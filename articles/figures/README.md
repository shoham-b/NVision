# Article figures

Put every figure of `nvision_paper.tex` in this folder (subfolders are fine). The same folder sits next to
`main.tex` on Overleaf, so one path works in both places.

`scripts/sync_overleaf.ps1` syncs these file types both ways: `.pdf .png .jpg .jpeg .eps .tex .tikz`.
Anything else here (like this README) stays local.

## Using a figure

Image (PDF is best for vector diagrams; PNG/JPG for photos). `\graphicspath` already points here, so give the
name only:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{levels}   % figures/levels.pdf
\caption{...}
\label{fig:levels}
\end{figure}
```

TikZ drawn in its own file (`figures/sequences.tex` holding just the `tikzpicture`):

```latex
\begin{figure}[t]
\centering
\input{figures/sequences}
\caption{...}
\label{fig:sequences}
\end{figure}
```

Notes:
- Column width in the two-column `reprint` layout is about 8.6 cm; use `figure*` and `\textwidth` for a full-width one.
- Add any TikZ library you need to the preamble of `nvision_paper.tex` (`\usetikzlibrary{...}`).
- Don't use the TikZ `external` library or `\write18`: it needs shell-escape.
- Sync: deleting a figure on one side does not delete it on the other; remove it on both.
- Added a figure on Overleaf? `scripts/sync_overleaf.ps1 pull` brings it down too (commit local work first:
  pull overwrites the local copies).
