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

        // Re-execute inline scripts (innerHTML doesn't execute scripts)
        var scripts = root.querySelectorAll('script');
        scripts.forEach(function(oldScript) {
            var newScript = document.createElement('script');
            if (oldScript.src) {
                newScript.src = oldScript.src;
            } else {
                newScript.textContent = oldScript.textContent;
            }
            oldScript.parentNode.replaceChild(newScript, oldScript);
        });
    } catch (e) {
        console.error('Failed to load chat widget:', e);
    }
})();