//go:build !(darwin && arm64)

package objc

// Available reports whether this build can call the Objective-C runtime.
const Available = false

var msgSendAddr, getClassAddr, selRegisterNameAddr, poolPushAddr, poolPopAddr uintptr

// call returns 0: there is no Objective-C runtime on this platform.
func call(fn uintptr, args ...uintptr) uintptr { return 0 }

// MTLCreateSystemDefaultDevice returns 0: there is no Metal on this
// platform.
func MTLCreateSystemDefaultDevice() ID { return 0 }
