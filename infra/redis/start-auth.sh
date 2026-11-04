#!/bin/sh
# Keep the Redis machine secret out of process arguments and source control.
set -eu
: "${MUSICGEN_REDIS_PASSWORD:?Set MUSICGEN_REDIS_PASSWORD to a random URL-safe secret}"
case "$MUSICGEN_REDIS_PASSWORD" in
  *[!A-Za-z0-9_-]*)
    printf '%s\n' 'MUSICGEN_REDIS_PASSWORD must contain only letters, digits, underscores and hyphens.' >&2
    exit 1
    ;;
esac
if [ "${#MUSICGEN_REDIS_PASSWORD}" -lt 32 ]; then
  printf '%s\n' 'MUSICGEN_REDIS_PASSWORD must contain at least 32 characters.' >&2
  exit 1
fi
umask 077
printf 'user default off\nuser cadence on >%s ~* &* +@all\n' "$MUSICGEN_REDIS_PASSWORD" > /data/users.acl
exec redis-server /usr/local/etc/redis/redis.conf --aclfile /data/users.acl
