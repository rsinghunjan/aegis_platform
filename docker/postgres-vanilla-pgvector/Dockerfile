FROM postgres:16

# Install build deps + pgvector
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    postgresql-server-dev-16 \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
 && cd /tmp/pgvector \
 && make \
 && make install \
 && rm -rf /tmp/pgvector

# (optional) keep default entrypoint/cmd from postgres image
