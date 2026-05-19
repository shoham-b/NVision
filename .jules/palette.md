## 2024-05-06 - Palette: Add ARIA roles to tab components
**Learning:** Adding WAI-ARIA tab attributes (role="tablist", "tab", "tabpanel") to custom interactive UI elements significantly improves screen reader navigation and clearly defines structure. To test these dynamic properties in isolation, we can mock `window.NVISION_BOOTSTRAP = Promise.resolve();` and check the rendered `role` and `aria-selected` attributes using Playwright evaluation expressions (`page.evaluate`).
**Action:** Always add appropriate roles and state-attributes to custom interactive components that emulate native form fields or layout controls (like tabs). When testing dynamic vanilla JS components that expect backend bootstrapped properties, create minimal mock HTML files that set those variables before the script runs.

## 2026-05-09 - Ensure all `iframe` elements have accessible titles
**Learning:** In visual-heavy reporting tools, iframes are frequently used to embed various types of plots and charts (e.g., Plotly graphs, Bayesian posterior visualizations). Screen reader users rely on the `title` attribute of the `iframe` to understand the content or purpose of the embedded document before deciding to interact with it. Without a `title`, screen readers often just announce "iframe" or the URL, which provides no context for the user.
**Action:** When adding or maintaining data visualizations embedded via `iframe`, always include a descriptive `title` attribute (e.g., `title="Parameter convergence plot"`). For hidden utility iframes (like background data loaders), include `title` and set `aria-hidden="true"`.
## 2026-05-13 - Segmented Controls and ARIA Keyboard Navigation
**Learning:** Native `role="radiogroup"` combined with `role="radio"` elements does not give users standard keyboard navigation implicitly. It requires explicit arrow key navigation using javascript and a roving tabindex to correctly shift focus without requiring tabbing through every single item.
**Action:** When implementing custom segmented controls with `role="radio"`, attach `keydown` listeners to cycle through nodes using `ArrowLeft`/`ArrowUp` and `ArrowRight`/`ArrowDown`, and manipulate `tabindex` to `0` / `-1` appropriately.

## 2024-05-14 - Tab Keyboard Navigation
**Learning:** Elements with `role="tab"` should use a "roving tabindex" strategy where only the active tab has `tabindex="0"` and inactive tabs have `tabindex="-1"`. Navigation between tabs within the same `role="tablist"` should be handled using arrow keys to improve keyboard accessibility for screen reader users.
**Action:** When implementing custom tab UI elements, ensure appropriate keydown listeners are attached for arrow keys, dynamically updating `tabindex` and focusing the newly selected tab.
## 2026-05-16 - Make dynamic help icons keyboard accessible
**Learning:** Dynamically generated HTML (like metric cards in `app.js`) often drops accessibility attributes if not explicitly included in the string templates. Here, help icon `<span>` elements lacked `tabindex="0"`, making their tooltips inaccessible to keyboard users navigating via Tab.
**Action:** Always ensure that interactive or tooltip-triggering elements rendered via JavaScript strings include `tabindex="0"` and appropriate CSS focus states (`:focus-visible`) to maintain accessibility.
## 2024-05-17 - Keyboard and Screen Reader Accessibility for Custom Tooltips and Dynamic Notifications
**Learning:** Custom UI elements like tooltip icons (e.g., `<span class="help-icon-visible">?</span>`) built with non-interactive elements do not receive keyboard focus by default, hiding their informative titles from keyboard-only and screen reader users. Additionally, dynamically inserted DOM elements (like error messages or status notifications) are ignored by screen readers unless explicitly marked with live region roles (`role="alert"` or `role="status" aria-live="polite"`).
**Action:** When adding custom tooltips using `<span>` or `<div>`, always include `tabindex="0"` to enable keyboard focus. When dynamically injecting error messages or status updates into the DOM via JavaScript, always set the appropriate `role` attribute (`alert` for errors, `status` for updates) so assistive technologies can announce them to the user.

## 2026-05-19 - Adding title tooltips to disabled repeat buttons
**Learning:** Native OS tooltips (via the `title` attribute) are effective for explaining disabled control states in this app, but they cannot be reliably captured in headless Playwright screenshots.
**Action:** When testing UI improvements that rely on `title` attributes, verify them programmatically by asserting `locator.get_attribute("title")` instead of relying solely on visual screenshot tests.
