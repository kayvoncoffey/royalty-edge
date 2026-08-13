# secrets/

`cookie.txt` goes here. It is gitignored.

Export it from the browser while logged in to Royalty Exchange:

1. DevTools -> Network -> Fetch/XHR
2. Right-click any request to auctions.royaltyexchange.com -> Copy -> Copy as cURL
3. Pull the value of `-H 'cookie: ...'` out of the command
4. Paste into `secrets/cookie.txt` (the leading `cookie:` is tolerated)

Or set `RE_COOKIE` in the environment instead, which takes precedence.

Session cookies expire. When `harvest` aborts with an auth failure, re-export
and resume -- the queue is preserved.
