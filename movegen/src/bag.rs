use rand::{Rng, seq::SliceRandom};

/// A shuffled collection from which items are drawn and removed.
///
/// Restocking replaces the existing contents with a newly shuffled collection.
#[derive(Debug, Default)]
pub struct Bag<T> {
    items: Vec<T>,
}

impl<T> Bag<T> {
    /// Creates a bag containing `items` in random order.
    pub fn new<R: Rng + ?Sized>(mut items: Vec<T>, rng: &mut R) -> Self {
        items.shuffle(rng);
        Bag { items }
    }

    /// Creates a bag containing `items` in the supplied draw order.
    pub fn from_items(items: Vec<T>) -> Self {
        Bag { items }
    }

    /// Replaces the contents with `items` in random order.
    ///
    /// Items previously in the bag are discarded.
    pub fn restock<R: Rng + ?Sized>(&mut self, mut items: Vec<T>, rng: &mut R) {
        items.shuffle(rng);
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
