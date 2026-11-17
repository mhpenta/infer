package cuda

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

var soakTime = flag.Duration("cuda.soak", 2*time.Minute, "how long TestSoak runs")

// threads returns the thread IDs of process pid ("self" for this one),
// split into Go's threads and others. Go's threads keep the process's name;
// libcuda names its own (cuda-EvtHandlr, cuda00002000009).
func threads(pid string) (goThreads, foreign []int, err error) {
	dir := filepath.Join("/proc", pid, "task")
	comm, err := os.ReadFile(filepath.Join("/proc", pid, "comm"))
	if err != nil {
		return nil, nil, err
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	for _, e := range ents {
		var tid int
		if _, err := fmt.Sscan(e.Name(), &tid); err != nil {
			continue
		}
		name, err := os.ReadFile(filepath.Join(dir, e.Name(), "comm"))
		if err != nil {
			continue // the thread exited
		}
		if bytes.Equal(name, comm) {
			goThreads = append(goThreads, tid)
		} else {
			foreign = append(foreign, tid)
		}
	}
	return goThreads, foreign, nil
}

// TestLockedThreadExit has goroutines lock their thread, use the device and
// return without unlocking, so the runtime ends each thread: mstart returns
// to fakecgo's threadentry, which returns to glibc, which runs libcuda's
// thread-exit destructors for the context made current there. It checks
// every launch, then that the threads really went away.
func TestLockedThreadExit(t *testing.T) {
	d := openDevice(t)
	f := loadVadd(t, d)
	before, _, err := threads("self")
	if err != nil {
		t.Fatal(err)
	}

	const rounds, perRound = 50, 8
	var ended atomic.Int64
	for r := range rounds {
		var wg sync.WaitGroup
		errs := make(chan error, perRound)
		for g := range perRound {
			wg.Go(func() {
				runtime.LockOSThread() // never unlocked: the thread exits with the goroutine
				if err := vectorAdd(d, f, 1000+g, uint64(r*perRound+g)); err != nil {
					errs <- err
				}
				ended.Add(1)
			})
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Fatalf("round %d: %v", r, err)
		}
	}

	// Threads end just after their goroutines do. Of the rounds*perRound
	// started, at most a few dozen may still be alive as idle Ms.
	limit := len(before) + 2*perRound
	var after []int
	for start := time.Now(); time.Since(start) < 5*time.Second; time.Sleep(20 * time.Millisecond) {
		if after, _, err = threads("self"); err != nil {
			t.Fatal(err)
		}
		if len(after) <= limit {
			break
		}
	}
	t.Logf("%d goroutines ended on locked threads; Go threads %d before, %d after", ended.Load(), len(before), len(after))
	if len(after) > limit {
		t.Errorf("%d Go threads remain, want at most %d: locked threads are not exiting", len(after), limit)
	}
	// The device must still work from a fresh thread.
	if err := vectorAdd(d, f, 4096, 7); err != nil {
		t.Fatal(err)
	}
}

// TestSoak launches kernels from several goroutines for -cuda.soak (two
// minutes by default) with the CPU profiler on, so SIGPROF arrives
// throughout, and also sends SIGPROF and SIGWINCH straight to libcuda's
// threads: on a thread Go did not start, the runtime's signal handler runs
// with no g, which is the path most likely to break without cgo. Some
// goroutines end on locked threads, so threads come and go throughout.
func TestSoak(t *testing.T) {
	if testing.Short() {
		t.Skip("soak test: runs without -short")
	}
	d := openDevice(t)
	f := loadVadd(t, d)
	if err := pprof.StartCPUProfile(io.Discard); err == nil {
		defer pprof.StopCPUProfile()
	} // otherwise -cpuprofile already has the profiler running

	var launches, threadExits, signals atomic.Int64
	stop := make(chan struct{})
	time.AfterFunc(*soakTime, func() { close(stop) })
	stopped := func() bool {
		select {
		case <-stop:
			return true
		default:
			return false
		}
	}

	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for w := range 6 {
		wg.Go(func() {
			for i := 0; !stopped(); i++ {
				n := 1 + (w*7919+i*104729)%300000
				if i%50 == 49 {
					// Use the device from a locked thread that then exits.
					done := make(chan error)
					go func() {
						runtime.LockOSThread()
						done <- vectorAdd(d, f, n, uint64(w<<32|i))
					}()
					if err := <-done; err != nil {
						errs <- err
						return
					}
					threadExits.Add(1)
				} else if err := vectorAdd(d, f, n, uint64(w<<32|i)); err != nil {
					errs <- err
					return
				}
				launches.Add(1)
			}
		})
	}
	wg.Go(func() {
		pid := os.Getpid()
		for !stopped() {
			_, foreign, err := threads("self")
			if err != nil {
				errs <- err
				return
			}
			for _, tid := range foreign {
				for _, sig := range []syscall.Signal{syscall.SIGPROF, syscall.SIGWINCH} {
					if syscall.Tgkill(pid, tid, sig) == nil {
						signals.Add(1)
					}
				}
			}
			time.Sleep(2 * time.Millisecond)
		}
	})
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	goThreads, foreign, _ := threads("self")
	t.Logf("%v: %d launches, %d on threads that then exited, %d signals sent to libcuda's threads; now %d Go and %d libcuda threads",
		*soakTime, launches.Load(), threadExits.Load(), signals.Load(), len(goThreads), len(foreign))
	if signals.Load() == 0 {
		t.Error("found no libcuda threads to signal")
	}
	if err := d.Synchronize(); err != nil {
		t.Fatal(err)
	}
}

const interruptChildEnv = "INFER_CUDA_INTERRUPT_CHILD"

// TestInterruptChild is the process TestInterrupt interrupts. It launches
// kernels until killed.
func TestInterruptChild(t *testing.T) {
	if os.Getenv(interruptChildEnv) == "" {
		t.Skip("run by TestInterrupt")
	}
	d := openDevice(t)
	f := loadVadd(t, d)
	for i := 0; ; i++ {
		if err := vectorAdd(d, f, 1<<16, uint64(i)); err != nil {
			t.Fatal(err)
		}
		if i == 20 {
			fmt.Println("ready")
		}
	}
}

// TestInterrupt is Ctrl-C during a run: it starts a child launching kernels
// and sends it SIGINT, either to the process, as a terminal does, or to one
// of libcuda's threads, where the runtime's handler runs with no g. Either
// way the child must die from SIGINT, as any Go program that does not catch
// it does, promptly and without a crash report.
func TestInterrupt(t *testing.T) {
	if testing.Short() {
		t.Skip("starts child processes: runs without -short")
	}
	openDevice(t) // skip when there is no device
	for _, target := range []string{"process", "libcuda thread"} {
		for run := range 3 {
			t.Run(fmt.Sprintf("%s/%d", strings.ReplaceAll(target, " ", "_"), run), func(t *testing.T) {
				interrupt(t, target)
			})
		}
	}
}

func interrupt(t *testing.T, target string) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestInterruptChild$")
	cmd.Env = append(os.Environ(), interruptChildEnv+"=1")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Process.Kill()
	ready := make(chan bool, 1)
	go func() {
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			if sc.Text() == "ready" {
				ready <- true
				break
			}
		}
		io.Copy(io.Discard, stdout)
	}()
	select {
	case <-ready:
	case <-time.After(30 * time.Second):
		t.Fatalf("child never became ready; stderr:\n%s", stderr.String())
	}

	pid := cmd.Process.Pid
	if target == "process" {
		err = cmd.Process.Signal(os.Interrupt)
	} else {
		_, foreign, terr := threads(fmt.Sprint(pid))
		if terr != nil || len(foreign) == 0 {
			t.Fatalf("no libcuda threads in the child: %v", terr)
		}
		err = syscall.Tgkill(pid, foreign[0], syscall.SIGINT)
	}
	if err != nil {
		t.Fatal(err)
	}
	sent := time.Now()

	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	select {
	case err = <-exited:
	case <-time.After(10 * time.Second):
		t.Fatalf("child still running 10s after SIGINT; stderr:\n%s", stderr.String())
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("child exited with %v, want death by SIGINT; stderr:\n%s", err, stderr.String())
	}
	ws := exitErr.Sys().(syscall.WaitStatus)
	if !ws.Signaled() || ws.Signal() != syscall.SIGINT {
		t.Errorf("child: %v, want death by SIGINT", exitErr)
	}
	for _, bad := range []string{"fatal error", "panic", "SIGSEGV", "goroutine "} {
		if strings.Contains(stderr.String(), bad) {
			t.Errorf("child's stderr has %q:\n%s", bad, stderr.String())
			break
		}
	}
	t.Logf("SIGINT to the %s: %v after %v", target, exitErr, time.Since(sent).Round(time.Millisecond))
}
