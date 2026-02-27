// dep/path_state.rs — Interior-mutable state for A* pathfinding.
//
// This companion file exists because the .hom codegen wraps all variable
// arguments with `.clone()`.  For plain `Vec<T>` arguments (like the std
// `push` function), `.clone()` pushes to a temporary that is immediately
// dropped — the mutation is silently lost.
//
// Using `Rc<RefCell<Vec<_>>>` as the backing store means that `.clone()`
// produces a cheap reference-count bump to the SAME underlying data, so
// mutations through any clone are immediately visible to all handles.
//
// .hom modules import this via `use path_state` and use:
//
//   Point — simple (x, y) coordinate pair (avoids layout_types.hom import):
//     point_new(x, y) -> Point
//
//   Position encoding (grid width required for encode/decode):
//     pos_to_key(x, y, width) -> i32        flat row-major index
//     key_to_x(key, width)    -> i32        column from flat index
//     key_to_y(key, width)    -> i32        row    from flat index
//     key_to_str(key)         -> String     for heap item storage
//     str_to_key(s)           -> i32        String (not &str) input
//
//   CostData — Rc<RefCell<Vec<i32>>>, initialised to -1 (= unvisited):
//     cost_data_new(size)             -> CostData
//     cost_data_set(d, idx, val)
//     cost_data_get(d, idx)           -> i32   (-1 if out-of-bounds)
//
//   PointList — Rc<RefCell<Vec<(i32,i32)>>>, accumulates path points:
//     point_list_new()                -> PointList
//     point_list_push(pl, x, y)
//     point_list_len(pl)              -> i32
//     point_list_get_x(pl, idx)       -> i32
//     point_list_get_y(pl, idx)       -> i32
//     point_list_copy(pl)             -> PointList   (independent copy)
//     point_list_reversed(pl)         -> PointList   (reversed copy)

// ── Point ─────────────────────────────────────────────────────────────────────
// A 2D grid coordinate (column, row).
//
// Defined here (not via layout_types.hom) because the homun codegen generates
// `EnumName.Variant` (using dot) rather than `EnumName::Variant` (Rust scope
// resolution) for enum literal expressions in function bodies.  Importing
// layout_types.hom → types.hom transitively causes those broken enum
// initialiser functions to be included, which prevents compilation.
//
// Point is a plain struct with no enum dependencies, so it is safe to define
// in a pure .rs file.  The layout module converts between this Point and the
// one in layout_types.hom using the same field names.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

/// Create a new Point at the given (x, y) coordinates.
pub fn point_new(x: i32, y: i32) -> Point {
    Point { x, y }
}

// ── Position encoding ─────────────────────────────────────────────────────────
// Positions are stored as flat row-major indices: key = y * width + x.
// This lets us use a single Vec<i32> for cost and predecessor arrays
// instead of a HashMap<(i32,i32), i32>, avoiding the dict-mutation
// limitations of the .hom calling convention.

/// Convert (x, y) grid coordinates to a flat row-major index.
pub fn pos_to_key(x: i32, y: i32, width: i32) -> i32 {
    y * width + x
}

/// Extract the column (x) from a flat row-major index.
pub fn key_to_x(key: i32, width: i32) -> i32 {
    if width <= 0 { 0 } else { key % width }
}

/// Extract the row (y) from a flat row-major index.
pub fn key_to_y(key: i32, width: i32) -> i32 {
    if width <= 0 { 0 } else { key / width }
}

/// Format a flat key as a decimal String for heap item storage.
/// Accepts i32 directly — no &str / String mismatch.
pub fn key_to_str(key: i32) -> String {
    key.to_string()
}

/// Parse a decimal String back to a flat key.
/// Takes String (not &str) so the .hom codegen's `.clone()` call works.
pub fn str_to_key(s: String) -> i32 {
    s.trim().parse::<i32>().unwrap_or(-1)
}

// ── CostData ─────────────────────────────────────────────────────────────────
// A flat Vec<i32> wrapped in Rc<RefCell<...>> for interior mutability.
// All entries are initialised to -1, which is used as the "unvisited" sentinel.
// Predecessor entries use -2 to mean "start node (no predecessor)".

/// Interior-mutable i32 array, safe to clone in .hom's calling convention.
pub type CostData = std::rc::Rc<std::cell::RefCell<Vec<i32>>>;

/// Create a new CostData of `size` entries, all initialised to -1.
pub fn cost_data_new(size: i32) -> CostData {
    let n = size.max(0) as usize;
    std::rc::Rc::new(std::cell::RefCell::new(vec![-1i32; n]))
}

/// Write `val` to entry `idx`.  Silently ignores out-of-range indices.
pub fn cost_data_set(d: CostData, idx: i32, val: i32) {
    if idx >= 0 {
        if let Some(slot) = d.borrow_mut().get_mut(idx as usize) {
            *slot = val;
        }
    }
}

/// Read entry `idx`.  Returns -1 for out-of-range indices.
pub fn cost_data_get(d: CostData, idx: i32) -> i32 {
    if idx < 0 {
        return -1;
    }
    d.borrow().get(idx as usize).copied().unwrap_or(-1)
}

// ── PointList ─────────────────────────────────────────────────────────────────
// A Vec<(i32,i32)> wrapped in Rc<RefCell<...>>.
// Used to accumulate (x, y) waypoints while building the A* path.
// All .hom calls that pass a PointList argument emit `.clone()`, which is a
// cheap Rc reference-count bump — mutations through any clone are shared.

/// Interior-mutable list of (x, y) coordinate pairs.
pub type PointList = std::rc::Rc<std::cell::RefCell<Vec<(i32, i32)>>>;

/// Create a new empty PointList.
pub fn point_list_new() -> PointList {
    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))
}

/// Append the point (x, y) to the list.
pub fn point_list_push(pl: PointList, x: i32, y: i32) {
    pl.borrow_mut().push((x, y));
}

/// Return the number of points currently in the list.
pub fn point_list_len(pl: PointList) -> i32 {
    pl.borrow().len() as i32
}

/// Return the x-coordinate of the point at position `idx`.
/// Panics (index out of bounds) if idx is out of range — callers should
/// check `point_list_len` before accessing.
pub fn point_list_get_x(pl: PointList, idx: i32) -> i32 {
    pl.borrow()[idx as usize].0
}

/// Return the y-coordinate of the point at position `idx`.
pub fn point_list_get_y(pl: PointList, idx: i32) -> i32 {
    pl.borrow()[idx as usize].1
}

/// Return an independent copy of the list (same order).
pub fn point_list_copy(pl: PointList) -> PointList {
    std::rc::Rc::new(std::cell::RefCell::new(pl.borrow().clone()))
}

/// Return a new PointList with the elements in reverse order.
pub fn point_list_reversed(pl: PointList) -> PointList {
    let v: Vec<(i32, i32)> = pl.borrow().iter().cloned().rev().collect();
    std::rc::Rc::new(std::cell::RefCell::new(v))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── pos_to_key / key_to_x / key_to_y ────────────────────────────────────

    #[test]
    fn test_pos_encoding_roundtrip() {
        let width = 10;
        for y in 0..5i32 {
            for x in 0..10i32 {
                let key = pos_to_key(x, y, width);
                assert_eq!(key_to_x(key, width), x, "x mismatch at ({},{})", x, y);
                assert_eq!(key_to_y(key, width), y, "y mismatch at ({},{})", x, y);
            }
        }
    }

    #[test]
    fn test_pos_to_key_values() {
        // 5-wide grid: row 0 = keys 0..4, row 1 = keys 5..9
        assert_eq!(pos_to_key(0, 0, 5), 0);
        assert_eq!(pos_to_key(4, 0, 5), 4);
        assert_eq!(pos_to_key(0, 1, 5), 5);
        assert_eq!(pos_to_key(2, 3, 5), 17);
    }

    // ── key_to_str / str_to_key ──────────────────────────────────────────────

    #[test]
    fn test_key_str_roundtrip() {
        for k in [0i32, 1, 42, 999, -1] {
            assert_eq!(str_to_key(key_to_str(k)), k);
        }
    }

    #[test]
    fn test_str_to_key_accepts_string() {
        // Verify it takes String, not &str (mirrors .hom clone behaviour)
        assert_eq!(str_to_key(String::from("17")), 17);
        assert_eq!(str_to_key(String::from("-1")), -1);
        assert_eq!(str_to_key(String::from("bad")), -1);
    }

    // ── CostData ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cost_data_init_all_minus_one() {
        let d = cost_data_new(6);
        for i in 0..6i32 {
            assert_eq!(cost_data_get(d.clone(), i), -1);
        }
    }

    #[test]
    fn test_cost_data_set_get() {
        let d = cost_data_new(10);
        cost_data_set(d.clone(), 3, 7);
        assert_eq!(cost_data_get(d.clone(), 3), 7);
        assert_eq!(cost_data_get(d.clone(), 2), -1);
    }

    #[test]
    fn test_cost_data_clone_shares_data() {
        let d = cost_data_new(5);
        let d2 = d.clone();
        cost_data_set(d.clone(), 2, 99);
        assert_eq!(cost_data_get(d2, 2), 99);
    }

    #[test]
    fn test_cost_data_oob_returns_minus_one() {
        let d = cost_data_new(4);
        assert_eq!(cost_data_get(d.clone(), 10), -1);
        assert_eq!(cost_data_get(d.clone(), -1), -1);
    }

    // ── PointList ─────────────────────────────────────────────────────────────

    #[test]
    fn test_point_list_push_len() {
        let pl = point_list_new();
        assert_eq!(point_list_len(pl.clone()), 0);
        point_list_push(pl.clone(), 3, 4);
        assert_eq!(point_list_len(pl.clone()), 1);
        point_list_push(pl.clone(), 7, 2);
        assert_eq!(point_list_len(pl.clone()), 2);
    }

    #[test]
    fn test_point_list_get() {
        let pl = point_list_new();
        point_list_push(pl.clone(), 10, 20);
        point_list_push(pl.clone(), 30, 40);
        assert_eq!(point_list_get_x(pl.clone(), 0), 10);
        assert_eq!(point_list_get_y(pl.clone(), 0), 20);
        assert_eq!(point_list_get_x(pl.clone(), 1), 30);
        assert_eq!(point_list_get_y(pl.clone(), 1), 40);
    }

    #[test]
    fn test_point_list_clone_shares() {
        let pl = point_list_new();
        let pl2 = pl.clone();
        point_list_push(pl.clone(), 5, 6);
        assert_eq!(point_list_len(pl2), 1);
    }

    #[test]
    fn test_point_list_copy_independent() {
        let pl = point_list_new();
        point_list_push(pl.clone(), 1, 2);
        let copy = point_list_copy(pl.clone());
        point_list_push(pl.clone(), 3, 4);
        // copy was made before the second push — should still have 1 item
        assert_eq!(point_list_len(copy), 1);
    }

    #[test]
    fn test_point_list_reversed() {
        let pl = point_list_new();
        point_list_push(pl.clone(), 1, 10);
        point_list_push(pl.clone(), 2, 20);
        point_list_push(pl.clone(), 3, 30);
        let rev = point_list_reversed(pl.clone());
        assert_eq!(point_list_get_x(rev.clone(), 0), 3);
        assert_eq!(point_list_get_x(rev.clone(), 1), 2);
        assert_eq!(point_list_get_x(rev.clone(), 2), 1);
    }
}
