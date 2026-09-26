# Multi-stage build: Node.js build -> nginx:alpine serve, non-root, matching
# the main Cropwright image's convention (see repo-root Dockerfile).
FROM node:26-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# nginx-unprivileged: listens on 8080 as a non-root user out of the box,
# same base family as the main Cropwright image.
FROM nginxinc/nginx-unprivileged:1.30-alpine

LABEL org.opencontainers.image.title="Cropwright Documentation" \
      org.opencontainers.image.description="Docusaurus documentation site for Cropwright" \
      org.opencontainers.image.vendor="example-org LLC" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"

# The site is built for baseUrl /cropwright/ (site.config.ts), so it is served
# from that subpath, exactly as GitHub Pages will serve it.
COPY --from=build /app/build /usr/share/nginx/html/cropwright
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://127.0.0.1:8080/ || exit 1
