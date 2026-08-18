# PIPER frontend image (Batch 4). Multi-stage: build the static Vite
# bundle with Node, then serve it with a minimal nginx — the built
# output never needs Node at runtime.
#
# Build context is frontend/ itself (self-contained npm project):
#
#   docker build -t piper-frontend .
#
# IMPORTANT (Vite/Docker gotcha, documented per the instruction to
# include required environment configuration): VITE_API_BASE_URL is
# baked into the JS bundle at BUILD time, not read at container
# runtime. It must be a URL the USER'S BROWSER can reach — i.e. the
# backend's HOST-published port (e.g. http://localhost:8000) — never
# a Docker-internal service name like http://backend:8000, which only
# resolves inside the compose network, not from the browser running on
# the host. docker-compose.yml passes this as a build arg.

FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# nginx:alpine already includes busybox wget — no extra package needed.
# Uses 127.0.0.1, not "localhost": this container's /etc/hosts
# resolves "localhost" to ::1 first, but nginx's `listen 80;` (see
# nginx.conf) only binds IPv4, so a literal "localhost" healthcheck
# connects to [::1]:80 and gets a false "connection refused" —
# genuine bug found and fixed during Batch 4 verification (the app was
# actually serving requests correctly the whole time; only the
# healthcheck itself was wrong).
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD wget -q --spider http://127.0.0.1:80/ || exit 1
