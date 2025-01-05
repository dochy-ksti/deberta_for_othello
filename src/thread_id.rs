
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ThreadID {
    id: usize,
}

impl ThreadID {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
    pub fn id(&self) -> usize {
        self.id
    }
}