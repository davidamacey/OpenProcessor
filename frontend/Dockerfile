FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ENV PUBLIC_TRITON_API_URL=__RUNTIME__
ENV PUBLIC_API_PREFIX=__API_PREFIX__
RUN npm run build

# Runtime: nginx as a non-root user (uid 101) listening on 8080.
FROM nginxinc/nginx-unprivileged:1.30-alpine

ARG VERSION=0.1.0
ARG REVISION=unknown
LABEL org.opencontainers.image.title="Cropwright" \
      org.opencontainers.image.description="Keyboard-first, cluster-assisted web app for labeling image crops at scale. Frontend for OpenProcessor." \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

USER root
RUN apk upgrade --no-cache && rm -rf /usr/share/nginx/html/*
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.d/40-runtime-config.sh
# The runtime-config script rewrites placeholders in the bundle and in
# default.conf with `sed -i` at container start, as uid 101.
RUN chmod 0755 /docker-entrypoint.d/40-runtime-config.sh \
    && chown -R 101:0 /usr/share/nginx/html /etc/nginx/conf.d \
    && chmod -R g+w /usr/share/nginx/html /etc/nginx/conf.d
USER 101

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD wget -qO- http://127.0.0.1:8080/ || exit 1
