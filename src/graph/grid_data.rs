// dep/grid_data.rs — Interior-mutable flat boolean grid for A* pathfinding.
//
// This companion file exists because the .hom codegen wraps all variable
// arguments with `.clone()`.  For `&mut Vec<T>` parameters (like `std::push`),
// the clone pushes to a temporary that is immediately dropped — the mutation
// is lost.
//
// Using `Rc<RefCell<Vec<bool>>>` as the backing store means that `.clone()`
// produces a cheap reference-count bump, and both the original and the clone
// refer to the SAME underlying data.  Mutations through either handle are
// immediately visible to all.
//
// The grid is stored as a flat row-major array:
//   index = row * width + col
//
// .hom modules import this via `use grid_data` and call:
//   grid_data_new(width, height) -> GridData
//   grid_data_set(data, row, col, width, val)
//   grid_data_get(data, row, col, width) -> bool

// Note: Rc and RefCell are referenced via fully-qualified paths below
// (std::rc::Rc, std::cell::RefCell) rather than via `use` statements.
// This avoids E0252 "defined multiple times" when multiple dep .rs files
// are inlined into the same flat .rs file during standalone test compilation.

// ── GridData ──────────────────────────────────────────────────────────────────

/// An interior-mutable, clone-safe flat boolean grid.
/// Use this as the `data` field inside the .hom `OccupancyGrid` struct.
pub type GridData = std::rc::Rc<std::cell::RefCell<Vec<bool>>>;

/// Create a new GridData of size (width × height), all cells initialised to
/// `false` (i.e. free).
pub fn grid_data_new(width: i32, height: i32) -> GridData {
    let n = (width * height).max(0) as usize;
    std::rc::Rc::new(std::cell::RefCell::new(vec![false; n]))
}

/// Set the cell at (col, row) in a flat row-major grid of the given `width`.
/// No bounds checking — callers are expected to check bounds first.
pub fn grid_data_set(data: GridData, row: i32, col: i32, width: i32, val: bool) {
    let idx = (row * width + col) as usize;
    data.borrow_mut()[idx] = val;
}

/// Get the value of the cell at (col, row) in a flat row-major grid.
/// No bounds checking — callers are expected to check bounds first.
pub fn grid_data_get(data: GridData, row: i32, col: i32, width: i32) -> bool {
    let idx = (row * width + col) as usize;
    data.borrow()[idx]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_data_new_all_false() {
        let d = grid_data_new(4, 3);
        for row in 0..3i32 {
            for col in 0..4i32 {
                assert!(!grid_data_get(d.clone(), row, col, 4));
            }
        }
    }

    #[test]
    fn test_grid_data_new_zero_size() {
        let d = grid_data_new(0, 0);
        assert_eq!(d.borrow().len(), 0);
    }

    #[test]
    fn test_grid_data_set_get() {
        let d = grid_data_new(5, 5);
        grid_data_set(d.clone(), 2, 3, 5, true);
        assert!(grid_data_get(d.clone(), 2, 3, 5));
        assert!(!grid_data_get(d.clone(), 2, 4, 5));
        assert!(!grid_data_get(d.clone(), 3, 3, 5));
    }

    #[test]
    fn test_grid_data_clone_shares_data() {
        let d = grid_data_new(3, 3);
        let d2 = d.clone();
        grid_data_set(d.clone(), 1, 1, 3, true);
        // d2 is a clone of d — they share the same Rc<RefCell<...>>
        assert!(grid_data_get(d2, 1, 1, 3));
    }

    #[test]
    fn test_grid_data_set_multiple_cells() {
        let d = grid_data_new(6, 4);
        grid_data_set(d.clone(), 0, 0, 6, true);
        grid_data_set(d.clone(), 1, 2, 6, true);
        grid_data_set(d.clone(), 3, 5, 6, true);
        assert!(grid_data_get(d.clone(), 0, 0, 6));
        assert!(grid_data_get(d.clone(), 1, 2, 6));
        assert!(grid_data_get(d.clone(), 3, 5, 6));
        // Unset cells remain false
        assert!(!grid_data_get(d.clone(), 0, 1, 6));
        assert!(!grid_data_get(d, 2, 3, 6));
    }
}
