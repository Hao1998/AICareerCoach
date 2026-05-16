(async function() {
    // Skip on login and register pages
    var path = window.location.pathname;
    if (path === '/login' || path === '/register') return;

    // Auth check - if not logged in, the history endpoint returns 401
    try {
        var authCheck = await fetch('/api/chat/history?limit=1');
        if (authCheck.status === 401 || authCheck.redirected) return;
    } catch (e) {
        return;
    }

    // Fetch widget HTML fragment
    try {
        var res = await fetch('/chat-widget');
        if (!res.ok) return;
        var html = await res.text();

        // Create container and inject
        var root = document.createElement('div');
        root.id = 'chat-widget-root';
        root.innerHTML = html;
        document.body.appendChild(root);

        // Re-execute scripts in order: load external scripts first, then inline.
        // forEach fires all at once — external scripts load async so inline runs
        // before io() is defined. Instead, chain them sequentially.
        var scripts = Array.from(root.querySelectorAll('script'));

        function runNext(i) {
            if (i >= scripts.length) return;
            var oldScript = scripts[i];
            var newScript = document.createElement('script');
            if (oldScript.src) {
                newScript.src = oldScript.src;
                newScript.onload = function() { runNext(i + 1); };
                newScript.onerror = function() { runNext(i + 1); };
                oldScript.parentNode.replaceChild(newScript, oldScript);
            } else {
                newScript.textContent = oldScript.textContent;
                oldScript.parentNode.replaceChild(newScript, oldScript);
                runNext(i + 1);
            }
        }

        runNext(0);
    } catch (e) {
        console.error('Failed to load chat widget:', e);
    }
})();