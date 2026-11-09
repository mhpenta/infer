// testdata is its own (empty) module so the Go toolchain leaves it out of
// the github.com/mhpenta/infer module download: the golden files are only
// needed to run the tests from a clone.
module github.com/mhpenta/infer/testdata

go 1.25.0
