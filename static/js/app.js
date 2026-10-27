
const INSIGHT_MAP = [
  {
    keys: ["attendance"],
    heading: "Attendance",
    type: "warning",
    explain: "Regular class participation keeps your predicted performance on track."
  },
  {
    keys: ["study"],
    heading: "Study Focus",
    type: "recommend",
    explain: "Consistent study time can help improve your predicted performance."
  },
  {
    keys: ["internet"],
    heading: "Digital Balance",
    type: "action",
    explain: "Reducing unnecessary internet usage can give you more focused study time."
  },
  {
    keys: ["sleep"],
    heading: "Wellness",
    type: "action",
    explain: "Aim for a consistent sleep schedule to support concentration."
  },
  {
    keys: ["assignment"],
    heading: "Assignments",
    type: "warning",
    explain: "Strengthening assignment quality can lift your overall score."
  },
  {
    keys: ["excellent", "great", "maintain", "well done", "good work"],
    heading: "Strength",
    type: "positive",
    explain: "Keep the habits that are already supporting strong results."
  },
  {
    keys: ["distract", "avoid"],
    heading: "Focus",
    type: "warning",
    explain: "Limiting distractions helps protect study quality and attention."
  }
];

function categoryFromScore(score) {
  const value = Number(score) || 0;

  if (value >= 80) return { label: "Excellent", tone: "success" };
  if (value >= 65) return { label: "Good", tone: "info" };
  if (value >= 50) return { label: "Average", tone: "warning" };

  return { label: "Needs Improvement", tone: "danger" };
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

/* Format inline Markdown safely */
function formatInline(text) {
  let safe = escapeHtml(text);

  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/__(.+?)__/g, "<strong>$1</strong>");
  safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
  safe = safe.replace(/\*(.+?)\*/g, "<em>$1</em>");

  return safe;
}

/* Format AI answers into readable HTML */
function formatAiReply(text) {
  const lines = String(text || "").split("\n");

  let html = "";
  let inList = false;
  let inCode = false;
  let codeLines = [];

  function closeList() {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (line.startsWith("```")) {
      closeList();

      if (inCode) {
        html += `<pre><code>${codeLines.join("\n")}</code></pre>`;
        codeLines = [];
        inCode = false;
      } else {
        inCode = true;
      }

      continue;
    }

    if (inCode) {
      codeLines.push(escapeHtml(rawLine));
      continue;
    }

    if (!line) {
      closeList();
      continue;
    }

    const heading = line.match(/^#{1,6}\s+(.+)$/);

    if (heading) {
      closeList();
      html += `<h4>${escapeHtml(heading[1])}</h4>`;
      continue;
    }

    const bullet = line.match(/^(?:[-*•]|\d+[.)])\s+(.+)$/);

    if (bullet) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }

      html += `<li>${formatInline(bullet[1])}</li>`;
      continue;
    }

    closeList();
    html += `<p>${formatInline(line)}</p>`;
  }

  closeList();

  if (inCode) {
    html += `<pre><code>${codeLines.join("\n")}</code></pre>`;
  }

  return html || "<p>No response was generated.</p>";
}

/* Create a chat bubble */
function appendBubble(log, role, html) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.innerHTML = html;

  log.appendChild(div);
  log.scrollTop = log.scrollHeight;

  return div;
}

/* Dashboard insight parsing */
function parseAiPoints(reply) {
  if (!reply) return [];

  return String(reply)
    .split("\n")
    .map(line => line.replace(/^[\s•\-]+/, "").trim())
    .filter(Boolean)
    .slice(0, 4);
}

function classifyInsight(text) {
  const lower = text.toLowerCase();

  const match = INSIGHT_MAP.find(item =>
    item.keys.some(key => lower.includes(key))
  );

  if (match) {
    return {
      heading: match.heading,
      type: match.type,
      title: text,
      body: match.explain
    };
  }

  if (
    lower.includes("improve") ||
    lower.includes("increase") ||
    lower.includes("reduce")
  ) {
    return {
      heading: "Recommendation",
      type: "recommend",
      title: text,
      body: "A focused change here can improve your predicted academic outcome."
    };
  }

  return {
    heading: "Insight",
    type: "action",
    title: text,
    body: "Use this suggestion to refine your study plan and daily routine."
  };
}

function insightCardsHtml(reply) {
  const points = parseAiPoints(reply);

  if (!points.length) {
    return `<div class="alert alert-info">
      No AI insights yet. Ask the assistant after you make a prediction.
    </div>`;
  }

  return `<div class="grid grid-insights">
    ${points.map(point => {
      const item = classifyInsight(point);

      const badgeType =
        item.type === "positive" ? "success" :
        item.type === "warning" ? "warning" :
        item.type === "recommend" ? "info" : "neutral";

      return `<article class="insight ${item.type}">
        <div class="insight-icon">${escapeHtml(item.heading.slice(0, 2))}</div>
        <div>
          <span class="badge ${badgeType}">${escapeHtml(item.heading)}</span>
          <h4>${escapeHtml(item.title)}</h4>
          <p>${escapeHtml(item.body)}</p>
        </div>
      </article>`;
    }).join("")}
  </div>`;
}

/* AI error handling */
function isAiError(reply) {
  const text = String(reply || "").toLowerCase();

  return (
    text.includes("ai error") ||
    text.includes("not configured") ||
    text.includes("login first")
  );
}

function friendlyAiError(reply) {
  const text = String(reply || "").toLowerCase();

  if (text.includes("not configured")) {
    return "EduAI Assistant is unavailable until the AI service is configured. Predictions still work as usual.";
  }

  if (text.includes("please predict first")) {
    return "Make a prediction first so the assistant can personalize insights from your latest results.";
  }

  return "EduAI Assistant is temporarily unavailable. Please try again in a moment.";
}

/* Sidebar */
function toggleSidebar(force) {
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("sidebarOverlay");

  if (!sidebar) return;

  const open = typeof force === "boolean"
    ? force
    : !sidebar.classList.contains("open");

  sidebar.classList.toggle("open", open);

  if (overlay) {
    overlay.classList.toggle("open", open);
  }
}

/* Performance chart */
function initCharts() {
  const canvas = document.getElementById("performanceChart");

  if (!canvas || typeof Chart === "undefined") return;

  let data;

  try {
    data = JSON.parse(canvas.dataset.values || "[]");
  } catch (error) {
    console.error("Unable to read chart data:", error);
    return;
  }

  new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Study", "Attendance", "Internet", "Sleep", "Assignments"],
      datasets: [{
        data: [
          Math.min(100, (Number(data[1] || 0) / 8) * 100),
          Math.min(100, Number(data[0] || 0)),
          Math.max(0, Math.min(100, (Number(data[2] || 0) / 6) * 100)),
          Math.min(100, (Number(data[3] || 0) / 8) * 100),
          Math.min(100, Number(data[5] || 0))
        ],
        backgroundColor: [
          "#4f46e5",
          "#22c55e",
          "#f43f5e",
          "#f59e0b",
          "#38bdf8"
        ],
        borderWidth: 0
      }]
    },
    options: {
      cutout: "72%",
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 10,
            font: {
              family: "Inter",
              size: 12
            }
          }
        }
      }
    }
  });
}

/* Send a message to Flask */
async function requestChat(message) {
  const res = await fetch("/chatbot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message })
  });

  let data;

  try {
    data = await res.json();
  } catch {
    throw new Error("AI request failed");
  }

  const reply = data.response || data.reply || "";

  if (!res.ok) {
    throw new Error(reply || "AI request failed");
  }

  return String(reply);
}

/* Chat assistant */
function initAssistant() {
  const log = document.getElementById("chatLog");
  const input = document.getElementById("aiMsg");
  const sendBtn = document.getElementById("aiSend");

  if (!log || !input || !sendBtn) return;

  async function send() {
    const message = input.value.trim();

    if (!message || sendBtn.disabled) return;

    appendBubble(log, "user", escapeHtml(message).replace(/\n/g, "<br>"));

    input.value = "";
    sendBtn.disabled = true;

    const loading = document.createElement("div");
    loading.className = "bubble bot";
    loading.textContent = "EduAI is thinking...";

    log.appendChild(loading);
    log.scrollTop = log.scrollHeight;

    try {
      const reply = await requestChat(message);

      loading.remove();

      if (!reply.trim()) {
        appendBubble(
          log,
          "bot",
          "<p>I couldn't generate a response. Please try again.</p>"
        );
      } else if (
        isAiError(reply) ||
        reply.toLowerCase().includes("please predict first")
      ) {
        appendBubble(log, "bot", formatAiReply(friendlyAiError(reply)));
      } else {
        appendBubble(log, "bot", formatAiReply(reply));
      }
    } catch (err) {
      loading.remove();
      console.error("EduAI chat error:", err);

      appendBubble(
        log,
        "bot",
        formatAiReply(friendlyAiError(err.message))
      );
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener("click", send);

  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  /* Quick prompt chips, if present in your template */
  document.querySelectorAll("[data-prompt]").forEach(button => {
    button.addEventListener("click", () => {
      input.value = button.dataset.prompt || button.textContent.trim();
      send();
    });
  });
}

window.quickPrompt = function (message) {
  const input = document.getElementById("aiMsg");
  const sendBtn = document.getElementById("aiSend");

  if (!input || !sendBtn) {
    console.error("Assistant input or send button not found.");
    return;
  }

  input.value = message;
  sendBtn.click();
};


/* Dashboard AI insight cards */
function initDashboardInsights() {
  const box = document.getElementById("aiSuggestions");

  if (!box) return;

  if (box.dataset.hasResult !== "true") {
    box.innerHTML = `<div class="alert alert-info">
      Predict your performance to unlock personalized AI insights.
    </div>`;
    return;
  }

  box.innerHTML = `
    <div class="typing">
      <span></span><span></span><span></span>
    </div>`;

  requestChat(
    "Analyze my latest academic prediction. Return exactly 4 concise, useful recommendations, one per line, each beginning with •. Focus on attendance, study, digital balance, sleep, assignments, or maintaining strengths as relevant to my data."
  )
    .then(reply => {
      if (
        isAiError(reply) ||
        reply.toLowerCase().includes("please predict first")
      ) {
        box.innerHTML =
          `<div class="alert alert-info">${escapeHtml(friendlyAiError(reply))}</div>`;
        return;
      }

      box.innerHTML = insightCardsHtml(reply);
    })
    .catch(err => {
      box.innerHTML =
        `<div class="alert alert-error">${escapeHtml(friendlyAiError(err.message))}</div>`;
    });
}

/* Prediction form */
function initPredictForm() {
  const form = document.getElementById("predictForm");
  const button = document.getElementById("predictSubmit");

  if (!form || !button) return;

  form.addEventListener("submit", () => {
    button.disabled = true;
    button.textContent = "Predicting...";
  });
}

/* Initialize the page */
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initAssistant();
  initDashboardInsights();
  initPredictForm();
});
