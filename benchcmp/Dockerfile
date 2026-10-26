# Runs the MatMul comparison on whatever x86-64 machine hosts the container,
# single-threaded like a 1 vCPU cloud instance. Build from the repository root:
#
#   docker build --platform linux/amd64 -f benchcmp/Dockerfile -t infer-benchcmp .
#   docker run --rm infer-benchcmp > results.txt
#
# Results are only meaningful on real x86 hardware: under Rosetta or QEMU every
# implementation runs at roughly the same emulated speed.
FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
COPY benchcmp/go.mod benchcmp/go.sum benchcmp/
RUN cd benchcmp && go mod download
COPY . .
RUN cd benchcmp && CGO_ENABLED=0 go test -c -o /benchcmp .

FROM gcr.io/distroless/static-debian12
COPY --from=build /benchcmp /benchcmp
ENV GOMAXPROCS=1
ENTRYPOINT ["/benchcmp", "-test.run", ".", "-test.bench", ".", "-test.count", "6"]
