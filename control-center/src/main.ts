import App from "./App.svelte";
import { mount } from "svelte";

function formatStartupError(error: unknown) {
  if (error instanceof Error) {
    return error.stack || error.message;
  }
  return String(error);
}

function showStartupError(error: unknown) {
  console.error("gary4local startup error:", error);

  const target = document.getElementById("app");
  if (!target) return;

  target.innerHTML = "";

  const container = document.createElement("main");
  container.style.cssText = [
    "min-height: 100vh",
    "padding: 32px",
    "box-sizing: border-box",
    "background: #111318",
    "color: #f3f5f7",
    "font-family: system-ui, sans-serif",
  ].join(";");

  const title = document.createElement("h1");
  title.textContent = "gary4local could not finish starting";
  title.style.cssText = "font-size: 22px; margin: 0 0 12px;";

  const body = document.createElement("p");
  body.textContent =
    "The control center hit a startup error before the normal interface loaded.";
  body.style.cssText = "max-width: 720px; color: #c7ccd3; line-height: 1.5;";

  const details = document.createElement("pre");
  details.textContent = formatStartupError(error);
  details.style.cssText = [
    "max-width: 100%",
    "overflow: auto",
    "padding: 16px",
    "border: 1px solid #303642",
    "background: #171a21",
    "border-radius: 8px",
    "white-space: pre-wrap",
  ].join(";");

  container.append(title, body, details);
  target.appendChild(container);
}

window.addEventListener("error", (event) => {
  console.error("gary4local window error:", event.error || event.message);
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("gary4local unhandled rejection:", event.reason);
});

let app: ReturnType<typeof mount>;

try {
  const target = document.getElementById("app");
  if (!target) throw new Error("Missing #app mount target");

  app = mount(App, { target });
} catch (error) {
  showStartupError(error);
  throw error;
}

export default app;
