#!/usr/bin/env bash
# -------- CONFIG --------
DOCKERHUB_USER="prerana1205"
IMAGE_NAME="recolor-api"
VERSION="v6"

# -------- DERIVED --------
IMAGE_URI="${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"

# -------- COPY --------
cp -r ../inference ./inference
BUILD_DIR="."

# -------- BUILD --------
echo "🐳 Building image: ${IMAGE_URI}"

docker build --platform linux/amd64 -t "${IMAGE_URI}" "${BUILD_DIR}" 

# -------- PUSH --------
echo "🚀 Pushing image to Docker Hub"
docker push "${IMAGE_URI}"

# -------- CLEANUP --------
rm -rf ./inference

echo "✅ Done!"
echo "Image pushed: ${IMAGE_URI}"
