import "./styles/main.css";
import { mountApp } from "./app.js";
import { registerRoute, initRouter } from "./router.js";
import { renderChatPage } from "./pages/chat.js";
import { renderSearchPage } from "./pages/search.js";
import { renderDocumentsPage } from "./pages/documents.js";
import { renderAnalyticsPage } from "./pages/analytics.js";

mountApp(document.getElementById("app"));

registerRoute("/", renderChatPage);
registerRoute("/search", renderSearchPage);
registerRoute("/documents", renderDocumentsPage);
registerRoute("/analytics", renderAnalyticsPage);

initRouter();
