## 2024-05-04 - Initializing palette.md
## 2026-05-04 - Added global focus-visible styles
**Learning:** Keyboard accessibility and clear focus states are a crucial part of the user experience. Adding a single, global `:focus-visible` rule for interactive elements like `button`, `select`, `input`, and `a` ensures consistent and accessible behavior, rather than applying it randomly on specific elements (like `.segmented-control button`).
**Action:** Next time, I should look out for missing or inconsistent focus indicators when modifying the frontend UI, and establish global accessibility styles where possible.
## 2024-05-04 - Dynamic Tab Navigation Accessibility
**Learning:** When building dynamic components with JavaScript, it's easy to overlook ARIA properties (like `role`, `id`, `aria-controls`, `aria-selected`) that are naturally present in static HTML examples. In this case, the `tabpanel` containers had explicit `aria-labelledby` IDs pointing to non-existent dynamic tab elements.
**Action:** Next time I work on dynamic components (like tabs or menus), I'll make sure the JavaScript explicitly injects full a11y properties mirroring standard accessible static patterns, and also actively manages their state (like toggling `aria-selected` upon click).
