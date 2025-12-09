(function() {
    const API_URL = "http://127.0.0.1:5000/chat";
    let isExpanded = false;

    // ============================================
    // 1. ESTILOS CSS (Ajuste "Surgical" para listas)
    // ============================================
    const style = document.createElement('style');
    style.innerHTML = `
        /* --- Layout Principal --- */
        #rag-agent-container { position: fixed; bottom: 20px; right: 20px; z-index: 99999; font-family: 'Segoe UI', system-ui, sans-serif; display: flex; flex-direction: column; align-items: flex-end; }
        
        #rag-agent-button { width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #007bff, #0062cc); color: white; border: none; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.25); font-size: 28px; display: flex; align-items: center; justify-content: center; transition: transform 0.2s; }
        #rag-agent-button:hover { transform: scale(1.05); }

        #rag-agent-chat-window { display: none; width: 360px; height: 550px; background: #ffffff; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,0.15); flex-direction: column; overflow: hidden; margin-bottom: 16px; border: 1px solid #f0f0f0; transition: all 0.3s; }
        #rag-agent-chat-window.rag-expanded { width: 700px; height: 80vh; }

        .rag-header { background: #007bff; color: white; padding: 16px; font-weight: 600; display: flex; justify-content: space-between; align-items: center; }
        .rag-icon-btn { cursor: pointer; padding: 0 6px; font-size: 18px; }

        #rag-messages { flex: 1; padding: 20px; overflow-y: auto; background: #f4f6f9; display: flex; flex-direction: column; gap: 10px; }
        
        .message { padding: 12px 16px; border-radius: 12px; max-width: 88%; word-wrap: break-word; font-size: 14px; line-height: 1.5; position: relative; }
        .user-msg { background: #007bff; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
        .bot-msg { background: white; color: #1c1e21; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); border: 1px solid #e5e5e5; }
        
        /* --- CORRECCIÓN CRÍTICA DE LISTAS --- */
        .bot-msg ul { 
            margin: 0;           /* Eliminamos margen externo del navegador */
            padding-left: 20px;  /* Sangría estandar */
            margin-bottom: 8px;  /* Espacio controlado después de la lista */
        }
        .bot-msg li { 
            margin-bottom: 0px;  /* Cero espacio extra entre items */
            line-height: 1.4;    /* Altura de línea cómoda pero compacta */
        }
        /* Eliminar espacio si la lista es lo último en el mensaje */
        .bot-msg ul:last-child { margin-bottom: 0; }
        
        .bot-msg strong { color: #0056b3; font-weight: 700; }
        .rag-card-address { background-color: #f0f7ff; border-left: 3px solid #007bff; padding: 10px; margin: 8px 0; border-radius: 4px; display: flex; align-items: center; gap: 8px; font-size: 0.95em; color: #333; }
        .rag-card-address::before { content: '📍'; font-size: 18px; }

        .rag-input-area { padding: 12px; background: white; border-top: 1px solid #eee; display: flex; gap: 8px; }
        #rag-input { flex: 1; padding: 10px 14px; border: 1px solid #e4e6eb; border-radius: 20px; outline: none; background: #f8f9fa; font-size: 14px; }
        #rag-input:focus { border-color: #007bff; background: white; }
        #rag-send { padding: 0 18px; background: #007bff; color: white; border: none; border-radius: 20px; cursor: pointer; font-weight: 600; font-size: 13px; }

        .typing-indicator { display: flex; align-items: center; gap: 4px; padding: 12px 16px; min-width: 40px; }
        .dot { width: 6px; height: 6px; background-color: #b0b3b8; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
        .dot:nth-child(1) { animation-delay: -0.32s; } .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    `;
    document.head.appendChild(style);

    // ============================================
    // 2. LÓGICA
    // ============================================
    const container = document.createElement('div');
    container.id = 'rag-agent-container';
    container.innerHTML = `
        <div id="rag-agent-chat-window">
            <div class="rag-header">
                <span>👨‍🍳 Chef Virtual</span>
                <div><span id="rag-expand" class="rag-icon-btn">⤢</span><span id="rag-close" class="rag-icon-btn">✕</span></div>
            </div>
            <div id="rag-messages"><div class="message bot-msg">¡Hola! Estoy listo para cocinar.</div></div>
            <div class="rag-input-area"><input type="text" id="rag-input" placeholder="Pregunta algo..."><button id="rag-send">Enviar</button></div>
        </div>
        <button id="rag-agent-button">💬</button>
    `;
    document.body.appendChild(container);

    const msgsDiv = document.getElementById('rag-messages');
    const input = document.getElementById('rag-input');
    
    document.getElementById('rag-agent-button').onclick = () => {
        const win = document.getElementById('rag-agent-chat-window');
        win.style.display = win.style.display === 'flex' ? 'none' : 'flex';
        if(win.style.display === 'flex') input.focus();
    };
    document.getElementById('rag-close').onclick = () => document.getElementById('rag-agent-chat-window').style.display = 'none';
    document.getElementById('rag-expand').onclick = (e) => {
        document.getElementById('rag-agent-chat-window').classList.toggle('rag-expanded');
        e.target.innerText = document.getElementById('rag-agent-chat-window').classList.contains('rag-expanded') ? '⤡' : '⤢';
    };

    // --- PARSER CORREGIDO (Agrupación de Listas) ---
    function parseMarkdown(text) {
        let html = text;
        html = html.replace(/^> (.*$)/gim, '<div class="rag-card-address">$1</div>');
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        const lines = html.split('\n');
        let inList = false;
        let processedLines = []; // Array de líneas finales

        lines.forEach((line) => {
            const trimLine = line.trim();
            
            if (trimLine.startsWith('- ')) {
                const content = trimLine.substring(2);
                if (!inList) { 
                    inList = true; 
                    // INICIO DE BLOQUE DE LISTA: Todo en un solo string
                    processedLines.push(`<ul><li>${content}</li>`); 
                } else {
                    // SEGUIMOS EN LA LISTA: Añadimos al ÚLTIMO elemento del array (sin crear nueva línea)
                    processedLines[processedLines.length - 1] += `<li>${content}</li>`;
                }
            } else {
                if (inList) { 
                    inList = false; 
                    // CERRAMOS EL BLOQUE DE LISTA ANTERIOR
                    processedLines[processedLines.length - 1] += '</ul>';
                }
                // Añadimos la línea de texto normal
                processedLines.push(line);
            }
        });
        // Si terminamos y seguía abierta la lista, cerrarla
        if (inList) processedLines[processedLines.length - 1] += '</ul>';

        // Unimos todo con <br>
        html = processedLines.join('<br>');
        
        // --- LIMPIEZA FINAL DE ESPACIOS ---
        // Eliminar <br> antes y después de la lista para que el CSS controle el margen
        html = html.replace(/<br><ul>/g, '<ul>');
        html = html.replace(/<\/ul><br>/g, '</ul>');
        
        return html;
    }

    async function handleSend() {
        const text = input.value.trim();
        if (!text) return;

        const userDiv = document.createElement('div');
        userDiv.className = 'message user-msg';
        userDiv.innerText = text;
        msgsDiv.appendChild(userDiv);
        input.value = ''; input.disabled = true;

        const loader = document.createElement('div');
        loader.className = 'message bot-msg typing-indicator';
        loader.innerHTML = '<div class="dot"></div><div class="dot"></div>';
        msgsDiv.appendChild(loader);
        msgsDiv.scrollTop = msgsDiv.scrollHeight;

        try {
            const res = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            loader.remove();

            const botDiv = document.createElement('div');
            botDiv.className = 'message bot-msg';
            
            botDiv.style.opacity = 0;
            botDiv.innerHTML = parseMarkdown(data.response);
            msgsDiv.appendChild(botDiv);
            
            let op = 0;
            const timer = setInterval(() => {
                if (op >= 1) clearInterval(timer);
                botDiv.style.opacity = op;
                op += 0.1;
            }, 30);

        } catch (e) {
            loader.remove();
            const err = document.createElement('div');
            err.className = 'message bot-msg';
            err.style.color = 'red';
            err.innerText = "Error de conexión";
            msgsDiv.appendChild(err);
        } finally {
            input.disabled = false;
            input.focus();
            msgsDiv.scrollTop = msgsDiv.scrollHeight;
        }
    }

    document.getElementById('rag-send').onclick = handleSend;
    input.onkeypress = (e) => { if (e.key === 'Enter') handleSend(); };
})();