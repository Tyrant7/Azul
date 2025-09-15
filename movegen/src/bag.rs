use rand::{rng, seq::SliceRandom};

/// This struct is for handling a shuffled `Vec<T>` of items.
/// Items are removed from the bag when accessed and bags may be restocked at any time.
#[derive(Debug, Default)]
pub struct Bag<T> {
    items: Vec<T>,
}

impl<T> Bag<T> {
    /// Creates a new bag from `items` after shuffling them.
    pub fn new(mut items: Vec<T>) -> Self {
        items.shuffle(&mut rng());
        Bag { items }
    }

    /// Restocks the bag with the given `items` after shuffling them.  
    /// Items previously in this bag are not retained.
    pub fn restock(&mut self, mut items: Vec<T>) {
        items.shuffle(&mut rng());
        self.items = items;
    }

    /// Getter for the items in this bag.
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
