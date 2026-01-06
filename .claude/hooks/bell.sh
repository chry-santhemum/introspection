#!/bin/bash
for t in /dev/pts/*; do
  [ -w "$t" ] && printf '\a' > "$t" 2>/dev/null
done