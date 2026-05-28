## 2024-05-24 - Accessibility issue with iframe missing title attributes
**Learning:** Three iframes in the milestone plots container (`milestone-iframe-steps`, `milestone-iframe-err-fc`, `milestone-iframe-delta-fc`) were missing `title` attributes. This makes them inaccessible to screen readers.
**Action:** Add descriptive `title` attributes to all `iframe` elements, especially those rendering plots or data visualizations.
