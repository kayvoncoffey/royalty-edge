# secrets/

`cookie.txt` goes here. It is gitignored.

Export it from the browser while logged in to Royalty Exchange:

1. DevTools -> Network -> Fetch/XHR
2. Right-click any request to auctions.royaltyexchange.com -> Copy -> Copy as cURL
3. In the cURL output, find the line that starts with -b '...'
Copy only what's between the single quotes on that line.
That's the cookie string. Paste it into secrets/cookie.txt.
4. Paste into `secrets/cookie.txt` (the leading `cookie:` is tolerated)

Or set `RE_COOKIE` in the environment instead, which takes precedence.

Session cookies expire. When `harvest` aborts with an auth failure, re-export
and resume -- the queue is preserved.
