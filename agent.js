(function() {
    const API_URL = "http://127.0.0.1:5000/chat";

    // ============================================
    // 1. ESTILOS CSS (DISEÑO LIMPIO Y GOURMET)
    // ============================================
    const style = document.createElement('style');
    style.innerHTML = `
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

        /* Contenedor Principal */
        #rag-agent-container { position: fixed; bottom: 20px; right: 20px; z-index: 99999; font-family: 'Inter', sans-serif; display: flex; flex-direction: column; align-items: flex-end; }
        
        /* Botón Flotante */
        #rag-agent-button { width: 60px; height: 60px; border-radius: 50%; background: #2563eb; color: white; border: none; cursor: pointer; box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4); font-size: 28px; display: flex; align-items: center; justify-content: center; transition: transform 0.2s, background 0.2s; }
        #rag-agent-button:hover { transform: scale(1.05); background: #1d4ed8; }

        /* Ventana de Chat */
        #rag-agent-chat-window { display: none; width: 380px; height: 600px; background: #ffffff; border-radius: 12px; box-shadow: 0 12px 40px rgba(0,0,0,0.15); flex-direction: column; overflow: hidden; margin-bottom: 16px; border: 1px solid #e5e7eb; animation: slideIn 0.3s ease-out; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        /* Header */
        .rag-header { background: #ffffff; color: #1f2937; padding: 16px 20px; font-weight: 600; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f3f4f6; }
        .rag-header span { font-size: 16px; display: flex; align-items: center; gap: 8px; }
        .rag-icon-btn { cursor: pointer; color: #9ca3af; font-size: 18px; padding: 4px; transition: color 0.2s; }
        .rag-icon-btn:hover { color: #4b5563; }

        /* Área de Mensajes */
        #rag-messages { flex: 1; padding: 20px; overflow-y: auto; background: #f9fafb; display: flex; flex-direction: column; gap: 12px; scroll-behavior: smooth; }
        
        /* Burbujas de Mensaje */
        .message { padding: 12px 16px; border-radius: 12px; max-width: 85%; word-wrap: break-word; font-size: 14px; line-height: 1.6; position: relative; }
        
        .user-msg { background: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
        
        .bot-msg { background: white; color: #374151; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e5e7eb; }

        /* --- FORMATO RECETA (ESTILO LIMPIO) --- */
        .bot-msg h3 { margin: 8px 0 12px 0; color: #111827; font-size: 18px; font-weight: 700; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }
        .bot-msg strong { color: #2563eb; font-weight: 600; }
        
        /* Listas de Ingredientes (Bullets) */
        .bot-msg ul { list-style-type: none; padding: 0; margin: 8px 0 16px 0; }
        .bot-msg ul li { position: relative; padding-left: 20px; margin-bottom: 6px; color: #4b5563; }
        .bot-msg ul li::before { content: "•"; color: #2563eb; font-weight: bold; position: absolute; left: 0; }

        /* Listas de Pasos (Números) */
        .bot-msg ol { padding-left: 20px; margin: 8px 0 16px 0; color: #4b5563; }
        .bot-msg ol li { margin-bottom: 8px; padding-left: 5px; }
        
        /* Input Area */
        .rag-input-area { padding: 16px; background: white; border-top: 1px solid #f3f4f6; display: flex; gap: 10px; }
        #rag-input { flex: 1; padding: 12px 16px; border: 1px solid #e5e7eb; border-radius: 24px; outline: none; background: #f9fafb; font-size: 14px; transition: all 0.2s; }
        #rag-input:focus { border-color: #2563eb; background: white; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }
        #rag-send { padding: 0 20px; background: #2563eb; color: white; border: none; border-radius: 24px; cursor: pointer; font-weight: 600; font-size: 14px; transition: background 0.2s; }
        #rag-send:hover { background: #1d4ed8; }

        /* Loader */
        .typing-indicator { display: flex; align-items: center; gap: 4px; padding: 16px; width: fit-content; }
        .dot { width: 6px; height: 6px; background-color: #9ca3af; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
        .dot:nth-child(1) { animation-delay: -0.32s; } .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    `;
    document.head.appendChild(style);

    // ============================================
    // 2. CREACIÓN DEL DOM
    // ============================================
    const container = document.createElement('div');
    container.id = 'rag-agent-container';
    container.innerHTML = `
        <div id="rag-agent-chat-window">
            <div class="rag-header">
                <span>👨‍🍳 Chef Virtual</span>
                <span id="rag-close" class="rag-icon-btn">✕</span>
            </div>
            <div id="rag-messages"><div class="message bot-msg">¡Hola! ¿Qué se te antoja cocinar hoy? 🥘</div></div>
            <div class="rag-input-area"><input type="text" id="rag-input" placeholder="Pregunta por una receta..."><button id="rag-send">Enviar</button></div>
        </div>
        <button id="rag-agent-button">💬</button>
    `;
    document.body.appendChild(container);

    const msgsDiv = document.getElementById('rag-messages');
    const input = document.getElementById('rag-input');
    
    // Toggle Ventana
    document.getElementById('rag-agent-button').onclick = () => {
        const win = document.getElementById('rag-agent-chat-window');
        win.style.display = win.style.display === 'flex' ? 'none' : 'flex';
        if(win.style.display === 'flex') input.focus();
    };
    document.getElementById('rag-close').onclick = () => document.getElementById('rag-agent-chat-window').style.display = 'none';

    // ============================================
    // 3. PARSER DE MARKDOWN MEJORADO (LÓGICA CLAVE)
    // ============================================
    function parseMarkdown(text) {
        // 1. Sanitizar HTML básico para evitar inyección
        let html = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");

        // 2. Procesar Headers (### Titulo)
        html = html.replace(/### (.*$)/gim, '<h3>$1</h3>');

        // 3. Procesar Negritas (**texto**)
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // 4. Lógica de Listas (State Machine simplificada)
        // Convertimos el texto en lineas para procesar grupos
        let lines = html.split('\n');
        let output = [];
        let inList = false;
        let listType = null; // 'ul' o 'ol'

        lines.forEach(line => {
            let trim = line.trim();

            // Detectar Listas Desordenadas (- item)
            if (trim.startsWith('- ')) {
                if (!inList || listType !== 'ul') {
                    if (inList) output.push(`</${listType}>`); // Cerrar lista anterior si existía
                    output.push('<ul>');
                    inList = true;
                    listType = 'ul';
                }
                output.push(`<li>${trim.substring(2)}</li>`);
            }
            // Detectar Listas Numeradas (1. item)
            else if (/^\d+\.\s/.test(trim)) {
                if (!inList || listType !== 'ol') {
                    if (inList) output.push(`</${listType}>`);
                    output.push('<ol>');
                    inList = true;
                    listType = 'ol';
                }
                // Quitamos el número "1." del texto, el HTML <ol> pone los números
                output.push(`<li>${trim.replace(/^\d+\.\s/, '')}</li>`);
            }
            // Texto normal
            else {
                if (inList) {
                    output.push(`</${listType}>`);
                    inList = false;
                    listType = null;
                }
                if (trim.length > 0) output.push(trim + '<br>'); 
            }
        });

        if (inList) output.push(`</${listType}>`);

        return output.join('');
    }

    // ============================================
    // 4. ENVÍO DE MENSAJES
    // ============================================
    async function handleSend() {
        const text = input.value.trim();
        if (!text) return;

        // Mostrar usuario
        const userDiv = document.createElement('div');
        userDiv.className = 'message user-msg';
        userDiv.innerText = text;
        msgsDiv.appendChild(userDiv);
        input.value = ''; input.disabled = true;
        msgsDiv.scrollTop = msgsDiv.scrollHeight;

        // Mostrar loader
        const loader = document.createElement('div');
        loader.className = 'message bot-msg typing-indicator';
        loader.innerHTML = '<div class="dot"></div><div class="dot"></div>';
        msgsDiv.appendChild(loader);
        
        try {
            const res = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            
            loader.remove(); // Quitar loader

            // Crear respuesta del bot parseada
            const botDiv = document.createElement('div');
            botDiv.className = 'message bot-msg';
            botDiv.innerHTML = parseMarkdown(data.response); // Usamos el nuevo parser
            msgsDiv.appendChild(botDiv);

        } catch (e) {
            loader.remove();
            const err = document.createElement('div');
            err.className = 'message bot-msg';
            err.style.color = '#ef4444';
            err.innerText = "Error: No pude conectar con la cocina.";
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