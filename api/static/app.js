const homeScreen = document.getElementById('homeScreen');
const chatScreen = document.getElementById('chatScreen');
const chatBox    = document.getElementById('chatBox');

function fillInput(text) {
    document.getElementById('homeInput').value = text;
}

function goHome() {
    chatScreen.style.display = 'none';
    homeScreen.style.display = 'flex';
    chatBox.innerHTML = '';
}

async function startChat() {
    const text = document.getElementById('homeInput').value.trim();
    if (!text) return;
    homeScreen.style.display = 'none';
    chatScreen.style.display = 'flex';
    await ask(text);
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text  = input.value.trim();
    if (!text) return;
    input.value = '';
    await ask(text);
}

async function ask(text) {
    // User message
    const userMsg = document.createElement('div');
    userMsg.classList.add('message', 'user');
    userMsg.textContent = text;
    chatBox.appendChild(userMsg);

    // Loading indicator
    const loadingMsg = document.createElement('div');
    loadingMsg.classList.add('loading');
    loadingMsg.id = 'loadingMsg';
    loadingMsg.textContent = 'Thinking...';
    chatBox.appendChild(loadingMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        });
        const data = await response.json();
        document.getElementById('loadingMsg')?.remove();

        // Bot answer
        const botMsg = document.createElement('div');
        botMsg.classList.add('message', 'bot');
        botMsg.textContent = data.answer;
        chatBox.appendChild(botMsg);

        // Sources
        if (data.sources && data.sources.length > 0) {
            const toggle = document.createElement('button');
            toggle.classList.add('sources-toggle');
            toggle.textContent = 'Sources';
            chatBox.appendChild(toggle);

            const srcContent = document.createElement('div');
            srcContent.classList.add('sources-content');
            srcContent.textContent = data.sources.slice(0, 2).join('\n\n---\n');
            chatBox.appendChild(srcContent);

            toggle.addEventListener('click', () => {
                const isOpen = srcContent.classList.toggle('open');
                toggle.textContent = isOpen ? 'Hide Sources' : 'Sources';
            });
        }

        // Divider
        const div = document.createElement('div');
        div.classList.add('divider');
        chatBox.appendChild(div);

    } catch (e) {
        document.getElementById('loadingMsg')?.remove();
        const errMsg = document.createElement('div');
        errMsg.classList.add('message', 'bot');
        errMsg.textContent = 'Error connecting to the API.';
        chatBox.appendChild(errMsg);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

document.getElementById('homeInput').addEventListener('keypress', e => {
    if (e.key === 'Enter') startChat();
});

document.getElementById('chatInput').addEventListener('keypress', e => {
    if (e.key === 'Enter') sendMessage();
});
