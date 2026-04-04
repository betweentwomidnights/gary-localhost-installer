# gary4local Update Manifest

Phase 1 uses a tiny static manifest so `gary4local` can check for new releases without depending on the DGX inference stack.

Default manifest URL:

`https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json`

That gives us a clean first deployment path:

1. commit the manifest under `docs/updates/gary4local/stable.json`
2. enable GitHub Pages for the repo's `docs/` directory
3. point the desktop app at the stable URL above

If you want to move it onto your own domain later, the safest shape is still a static endpoint. Put Cloudflare or your domain in front of the same JSON file instead of tying update checks to the DGX docker network.

Recommended release flow:

1. build the NSIS installer
2. upload the installer to the GitHub release
3. compute `SHA256SUMS.txt`
4. update `docs/updates/gary4local/stable.json`
5. merge so the static manifest goes live
