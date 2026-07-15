# Keep this version aligned with the Playwright dependency in task repositories.
# The image supplies browsers and OS dependencies; each repository supplies its JS packages.
FROM mcr.microsoft.com/playwright:v1.61.1-noble

RUN corepack enable

# Cache the fixed V1 React/Vite fixture dependencies in the image so rollout
# installation remains reproducible with `--network none`. Projects with a
# different lockfile should build their own image from this Dockerfile.
COPY examples/frontend-expert/fixture-site/package.json \
     examples/frontend-expert/fixture-site/package-lock.json \
     /opt/autotrainer/frontend-deps/
RUN cd /opt/autotrainer/frontend-deps && npm ci --ignore-scripts

ENV CI=1 \
    HOME=/tmp/autotrainer-home \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /workspace

LABEL org.opencontainers.image.title="AutoTrainer frontend rollout runtime" \
      org.opencontainers.image.description="Network-disabled React/Vite/Playwright task execution" \
      org.opencontainers.image.licenses="Apache-2.0"
