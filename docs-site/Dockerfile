# Multi-stage build: Node.js build -> nginx:alpine serve, non-root.
FROM node:26-alpine@sha256:c6ad9e9c1d80f6eb607c26b4b0a132ff7a2344fad1c78ea3baa2a7d4c8de6d94 AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# nginx-unprivileged: listens on 8080 as a non-root user out of the box,
# same base family as the other project images.
FROM nginxinc/nginx-unprivileged:1.31-alpine@sha256:b5047b9b76845eac124358373c6852377c52e4b05834a47647e955dd91cb8f1e

LABEL org.opencontainers.image.title="OpenProcessor Documentation" \
      org.opencontainers.image.description="Docusaurus documentation site for OpenProcessor" \
      org.opencontainers.image.vendor="OpenProcessor Contributors" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"

# The site is built for baseUrl /OpenProcessor/ (site.config.ts), so it is served
# from that subpath, exactly as GitHub Pages will serve it.
COPY --from=build /app/build /usr/share/nginx/html/OpenProcessor
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["sh", "-c", "wget --quiet --tries=1 --spider http://127.0.0.1:8080/ || exit 1"]
