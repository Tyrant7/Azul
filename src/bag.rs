use rand::{rng, seq::SliceRandom};

use crate::protocol::ProtocolFormat;

#[derive(Debug)]
pub struct Bag<T> {
    items: Vec<T>,
}

impl<T> Bag<T> {
    pub fn new(mut items: Vec<T>) -> Self {
        items.shuffle(&mut rng());
        Bag { items }
    }

    pub fn restock(&mut self, mut items: Vec<T>) {
        items.shuffle(&mut rng());
        self.items = items;
    }
}

impl<T> Iterator for Bag<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop()
    }
}

impl<T> ProtocolFormat for Bag<T>
where
    T: ToString,
{
    fn fmt_human(&self) -> String {
        "".to_string()
    }

    fn fmt_uci_like(&self) -> String {
        self.items.iter().map(|t| t.to_string()).collect()
    }
}
