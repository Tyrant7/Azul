use rand::{rng, seq::SliceRandom};

/// A shuffled collection from which items are drawn and removed.
///
/// Restocking replaces the existing contents with a newly shuffled collection.
#[derive(Debug, Default)]
pub struct Bag<T> {
    items: Vec<T>,
}

impl<T> Bag<T> {
    /// Creates a bag containing `items` in random order.
    pub fn new(mut items: Vec<T>) -> Self {
        items.shuffle(&mut rng());
        Bag { items }
    }

    /// Replaces the contents with `items` in random order.
    ///
    /// Items previously in the bag are discarded.
    pub fn restock(&mut self, mut items: Vec<T>) {
        items.shuffle(&mut rng());
        self.items = items;
    }

    /// Returns the remaining items in their current draw order.
    ///
    /// The iterator implementation removes items from the end of this vector.
    pub fn items(&self) -> &Vec<T> {
        &self.items
    }
}

impl<T> Iterator for Bag<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop()
    }
}
