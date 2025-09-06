#![allow(unused)]

use std::io;

use crevice::std140::{self, AsStd140, WriteStd140};
// use crevice::std140;

fn main() {
    /* let x = Foo {
        max: 42,
        xs: vec![1, 2, 3, 4],
    };
    let buffer = Vec::new();
    let mut cursor = std::io::Cursor::new(buffer);
    x.write_std140(&mut std140::Writer::new(&mut cursor))
        .unwrap();
    let b = cursor.into_inner();
    dbg!(&x, b); */
}

#[cfg(false)]
#[derive(Debug, AsStd140)]
struct Bar {
    x: f32,
    y: f32,
}

#[derive(Debug)]
struct Foo {
    max: u32,
    xs: Vec<(f32, f32)>,
}

impl WriteStd140 for Foo {
    fn write_std140<W: io::Write>(&self, writer: &mut std140::Writer<W>) -> io::Result<usize> {
        let first_byte = writer.write(&self.max)?;
        // writer.write(self.xs.as_slice())?;
        writer.write_iter(self.xs.iter().copied().flat_map(|(l, r)| [l, r]));
        Ok(first_byte)
    }
}
