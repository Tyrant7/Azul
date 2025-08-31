use rand::{rng, seq::SliceRandom};

pub struct Bag<T> {
    items: Vec<T>,
}

impl<T> Bag<T> {
    pub fn new(mut items: Vec<T>) -> Self {
        items.shuffle(&mut rng());
        Bag { items }
    }

    pub fn shuffle(&mut self) {
        self.items.shuffle(&mut rng());
    }
}

impl<T> Iterator for Bag<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop()
    }
}
