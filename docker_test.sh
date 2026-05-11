docker build --target runtime -t test-runtime -f docker/worker/Dockerfile .
docker run --rm test-runtime
