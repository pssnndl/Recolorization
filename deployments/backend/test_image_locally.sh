# -------- CONFIG --------
DOCKERHUB_USER="prerana1205"
IMAGE_NAME="recolor-api"
VERSION="v4"

# -------- DERIVED --------
IMAGE_URI="${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"

docker run -d \
  --name recolor-api \
  -p 8000:8000 \
  "${IMAGE_URI}"


# docker run -it --rm --entrypoint /bin/sh "${IMAGE_URI}"
