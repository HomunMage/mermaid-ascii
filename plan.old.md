# text-graph

# 用 rust 重寫 graph-easy 的完整參考庫清單

## 🎯 你要做的事
DSL 文本輸入 → 解析 → 圖布局 → ASCII/Unicode 文字輸出

---

## 1. 現有同類工具（直接競品 / 靈感來源）

### DSL → ASCII 輸出（你要做的事）

| 工具 | 語言 | 輸入 | 輸出 | 備註 |
|------|------|------|------|------|
| **graph-easy** | Perl | 自定義 DSL / DOT | ASCII / Unicode | 原版，功能最完整但年久失修 |
| **mermaid-ascii** | Go | Mermaid 語法 | ASCII / Unicode | ⭐ **最接近你要做的事**，Alexander Grooff 寫的 |
| **beautiful-mermaid** | TypeScript | Mermaid 語法 | ASCII + SVG | 從 mermaid-ascii 移植到 TS，Craft 團隊開發 |
| **figurehead** | Rust | Mermaid 語法 | ASCII | Rust crate，將 Mermaid 轉 ASCII |
| **ascii-dag** | Rust | API（程式呼叫） | ASCII DAG | 零依賴，Sugiyama 布局，非常快 |
| **ADia** | CLI | 自定義語法 | ASCII sequence diagram | 專門做序列圖 |

### ASCII → SVG/PNG 輸出（反方向，但布局算法可參考）

| 工具 | 語言 | 輸入 | 輸出 | 備註 |
|------|------|------|------|------|
| **Svgbob** | Rust | ASCII art | SVG | Ivan Ceras 寫的，字符識別算法值得參考 |
| **GoAT** | Go | ASCII art | SVG | markdeep 的 Go 實現 |
| **asciitosvg** | Go | ASCII art | SVG | 另一個 ASCII → SVG |
| **Typograms** | JavaScript | ASCII art | SVG | Google 開發，瀏覽器原生 |
| **ditaa** | Java | ASCII art | PNG | 老牌工具 |

### DSL → SVG/PNG 輸出（不輸出 ASCII，但 DSL 設計和布局算法可參考）

| 工具 | 語言 | 輸入 | 輸出 | 備註 |
|------|------|------|------|------|
| **D2** | Go | D2 語法 | SVG/PNG | Terrastruct 開發，語法設計參考 |
| **Mermaid** | JavaScript | Mermaid 語法 | SVG | 事實標準，語法生態最大 |
| **Graphviz** | C | DOT 語法 | SVG/PNG/PDF | 圖布局算法的鼻祖 |
| **PlantUML** | Java | PlantUML 語法 | SVG/PNG | 功能最全的 UML 工具 |
| **Pikchr** | C | PIC-like 語法 | SVG | SQLite 作者 D. Richard Hipp 寫的，極簡 |
| **mermaid-rs-renderer** | Rust | Mermaid 語法 | SVG | 原生 Rust Mermaid 渲染器，比 mermaid-cli 快 500-1000x |
| **Kroki** | 多語言 | 多種 DSL | SVG/PNG | 統一 API 聚合多種圖表工具 |

### 互動式 ASCII 繪圖編輯器（非 DSL，但渲染技術可參考）

| 工具 | 平台 | 備註 |
|------|------|------|
| **ASCIIFlow** | Web | asciiflow.com，自由繪製 ASCII 圖 |
| **Monodraw** | macOS | 商業軟體，Mac 專用，品質最高 |
| **MonoSketch** | Web | 開源 ASCII 編輯器，Kotlin/JS |
| **ASCII Draw** | Linux (Flatpak) | GTK 桌面應用 |
| **Textik** | Web | textik.com，開源 |

---

## 2. Parser 解析器庫

你需要把 DSL 文本解析成 AST。

### Rust
| 庫 | 類型 | 備註 |
|-----|------|------|
| **pest** | PEG parser generator | ⭐ 推薦首選，寫 grammar 文件生成 parser |
| **nom** | Parser combinator | 函數式組合，適合手寫 parser |
| **logos** | Lexer generator | 超快的 tokenizer，搭配 parser 用 |
| **chumsky** | Parser combinator | 錯誤恢復能力強，API 友好 |
| **lalrpop** | LR(1) parser generator | 像 Yacc/Bison 的風格 |
| **tree-sitter** | Incremental parser | 適合需要即時編輯反饋的場景 |

### Go
| 庫 | 類型 | 備註 |
|-----|------|------|
| **participle** | Parser library | 用 struct tag 定義語法 |
| **pigeon** | PEG parser generator | Go 版 PEG |
| **antlr4-go** | ANTLR Go target | 經典 parser generator |

### C++
| 庫 | 類型 | 備註 |
|-----|------|------|
| **ANTLR4** | Parser generator | 業界標準，多語言 target |
| **Flex + Bison** | Lexer + Parser generator | C/C++ 經典組合 |
| **Boost.Spirit** | Parser combinator | C++ 模板元編程風格 |
| **lexy** | Parser combinator | 現代 C++17，錯誤處理好 |
| **PEGTL** | PEG parser | Header-only C++ PEG 庫 |

### OCaml
| 庫 | 類型 | 備註 |
|-----|------|------|
| **ocamllex + menhir** | Lexer + LR parser | ⭐ OCaml 生態最佳 parser 工具鏈 |
| **angstrom** | Parser combinator | 高性能 |
| **sedlex** | Unicode-aware lexer | 處理 Unicode 更好 |

---

## 3. 圖布局算法庫

把 AST 中的節點和邊計算出座標位置。

### 核心算法（需要理解的理論）
| 算法 | 用途 | 參考 |
|------|------|------|
| **Sugiyama (層次布局)** | ⭐ 最重要！有方向的流程圖布局 | 分層 → 減少交叉 → 座標分配 |
| **Force-directed** | 通用圖布局 | 彈簧模型，不太適合 ASCII |
| **Orthogonal routing** | 正交連線路徑 | ASCII 圖只能水平/垂直，這是剛需 |
| **Coffman-Graham** | 帶寬度限制的分層 | 控制圖的寬度 |

### Rust 實現
| 庫 | 備註 |
|-----|------|
| **layout-rs** | Graphviz DOT 的 Rust 實現，含 Sugiyama |
| **petgraph** | ⭐ Rust 最成熟的圖資料結構庫，拓撲排序、BFS/DFS |
| **ascii-dag** | 自帶 Sugiyama 布局的 ASCII DAG 渲染 |

### Go 實現
| 庫 | 備註 |
|-----|------|
| **mermaid-ascii 內建** | 自己實現的 grid-based 布局 |
| **gonum/graph** | Go 的圖資料結構和算法庫 |

### JavaScript 實現（可參考算法邏輯）
| 庫 | 備註 |
|-----|------|
| **dagre** | ⭐ Sugiyama 算法的 JS 標準實現，Mermaid 用這個 |
| **dagre-d3** | dagre + D3 渲染 |
| **ELK.js** | Eclipse Layout Kernel 的 JS 版，企業級布局 |
| **cytoscape.js** | 完整的圖可視化庫 |

### C/C++ 實現
| 庫 | 備註 |
|-----|------|
| **Graphviz (libgvc, libcgraph)** | ⭐ 布局算法的 C 參考實現 |
| **OGDF** | 開源圖繪製框架，學術級 |

---

## 4. ASCII / Unicode 字符渲染庫

把座標轉換成字符網格輸出。

### Rust
| 庫 | 用途 | 備註 |
|-----|------|------|
| **unicode-width** | ⭐ CJK 字符寬度計算 | ASCII 對齊必備 |
| **unicode-segmentation** | Unicode 文字分段 | 正確處理 grapheme cluster |
| **textwrap** | 文字自動換行 | 用於 box 內文字排版 |
| **ascii-canvas** | 2D 字符畫布 | LALRPOP 作者寫的，直接能用 |
| **ratatui** | TUI 框架 | 如果你想做互動式 TUI 版本 |
| **crossterm** | 終端控制 | 顏色、游標、跨平台 |
| **comfy-table** | 表格渲染 | 表格類圖表可以參考 |

### Python
| 庫 | 用途 | 備註 |
|-----|------|------|
| **Textual** | TUI 框架 | Will McGugan 寫的，Python TUI 首選 |
| **Rich** | 終端美化 | 表格、面板、box 渲染 |
| **wcwidth** | CJK 字符寬度 | Python 版的 unicode-width |
| **blessed / curses** | 終端控制 | 底層終端操作 |

### Go
| 庫 | 用途 | 備註 |
|-----|------|------|
| **bubbletea** | TUI 框架 | Charm 出品，Go TUI 首選 |
| **lipgloss** | 終端樣式 | Charm 出品，CSS-like 終端樣式 |
| **go-runewidth** | CJK 字符寬度 | Go 版的 unicode-width |
| **tcell** | 終端控制 | 底層跨平台終端庫 |

### C++
| 庫 | 用途 | 備註 |
|-----|------|------|
| **FTXUI** | TUI 框架 | 現代 C++ TUI，支持 flexbox 布局 |
| **ICU** | Unicode 處理 | 工業級 Unicode 庫 |
| **ncurses** | 終端控制 | Unix 終端標準 |

---

## 5. 測試 & 快照測試

確保輸出的 ASCII 圖不會 regression。

| 庫 | 語言 | 備註 |
|-----|------|------|
| **insta** | Rust | ⭐ 快照測試首選，適合比對 ASCII 輸出 |
| **expect_test** | Rust | Inline snapshot testing |
| **go-snaps** | Go | Go 版快照測試 |
| **jest (snapshot)** | JS/TS | JS 生態的快照測試 |
| **Catch2** | C++ | 支持 snapshot 的 C++ 測試框架 |

---

## 6. 建議的優先研讀順序

如果你要動手做，建議按這個順序研究：

```
1. mermaid-ascii (Go)          ← 最接近你要做的事，讀它的源碼
   github.com/AlexanderGrooff/mermaid-ascii

2. graph-easy (Perl)           ← 原版，理解功能全貌
   metacpan.org/pod/Graph::Easy

3. ascii-dag (Rust)            ← Sugiyama 布局 + ASCII 渲染的 Rust 參考實現
   crates.io/crates/ascii-dag

4. dagre (JS)                  ← 理解 Sugiyama 層次布局算法
   github.com/dagrejs/dagre

5. Svgbob (Rust)               ← 字符識別和渲染技術參考
   github.com/ivanceras/svgbob

6. D2 (Go)                     ← DSL 語法設計參考
   github.com/terrastruct/d2

7. Pikchr (C)                  ← 極簡 DSL 設計的典範
   pikchr.org

8. beautiful-mermaid (TS)      ← mermaid-ascii 的 TS 移植，看它怎麼擴展的
   github.com/lukilabs/beautiful-mermaid
```

---

## 7. 如果用 Rust，你的 Cargo.toml 大概長這樣

```toml
[dependencies]
# Parser
pest = "2"
pest_derive = "2"

# 圖資料結構 & 布局
petgraph = "0.6"

# Unicode 處理
unicode-width = "0.2"
unicode-segmentation = "1.10"
textwrap = "0.16"

# 終端輸出（可選，用於彩色輸出）
crossterm = "0.27"

# CLI
clap = { version = "4", features = ["derive"] }

[dev-dependencies]
insta = "1.34"
```