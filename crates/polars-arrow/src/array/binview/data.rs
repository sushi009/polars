use arrow_data::ArrayData;

use super::utf8_to::{binary_to_binview, utf8_to_utf8view};
use super::{BinaryViewArray, Utf8ViewArray};
use crate::array::{Array, Arrow2Arrow, BinaryArray, MutableBinaryValuesArray, Utf8Array};
use crate::offset::Offset;

/// This is highly inefficent as we are punting the conversion to `Utf8Array`
/// leading to double conversion
impl Arrow2Arrow for Utf8ViewArray {
    fn to_data(&self) -> ArrayData {
        let array = utf8view_to_utf8::<i64>(self);
        array.to_data()
    }

    fn from_data(data: &ArrayData) -> Self {
        let arr: arrow_array::StringViewArray = data.clone().into();
        let arr = arrow_cast::cast(&arr, &arrow_schema::DataType::LargeUtf8).expect("failed");
        let data = arr.to_data();
        let arr = Utf8Array::<i64>::from_data(&data);
        utf8_to_utf8view(&arr)
    }
}

/// This is highly inefficent as we are punting the conversion to `BinaryArray`
/// leading to double conversion
impl Arrow2Arrow for BinaryViewArray {
    fn to_data(&self) -> ArrayData {
        let arr = view_to_binary::<i64>(self);
        arr.to_data()
    }

    fn from_data(data: &ArrayData) -> Self {
        let arr: arrow_array::BinaryViewArray = data.clone().into();
        let arr = arrow_cast::cast(&arr, &arrow_schema::DataType::LargeBinary).expect("failed");
        let data = arr.to_data();
        let arr = BinaryArray::<i64>::from_data(&data);
        binary_to_binview(&arr)
    }
}

/// NOTE: Copied from polars-compute crate
pub fn utf8view_to_utf8<O: Offset>(array: &Utf8ViewArray) -> Utf8Array<O> {
    let array = array.to_binview();
    let out = view_to_binary::<O>(&array);

    let dtype = Utf8Array::<O>::default_dtype();
    unsafe {
        Utf8Array::new_unchecked(
            dtype,
            out.offsets().clone(),
            out.values().clone(),
            out.validity().cloned(),
        )
    }
}

/// NOTE: Copied from polars-compute crate
fn view_to_binary<O: Offset>(array: &BinaryViewArray) -> BinaryArray<O> {
    let len: usize = Array::len(array);
    let mut mutable = MutableBinaryValuesArray::<O>::with_capacities(len, array.total_bytes_len());
    for slice in array.values_iter() {
        mutable.push(slice)
    }
    let out: BinaryArray<O> = mutable.into();
    out.with_validity(array.validity().cloned())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_str_view_array() {
        use arrow_array::Array;
        let mut builder = arrow_array::builder::StringViewBuilder::new();
        builder.append_value("hello");
        builder.append_null();
        builder.append_value("world");
        let array = builder.finish();

        let data = array.to_data();
        let arr = Utf8ViewArray::from_data(&data);
        let mut x = arr.values_iter();
        assert_eq!(x.next(), Some("hello"));
        // TODO: Why is it not null
        assert_eq!(x.next(), Some(""));
        assert_eq!(x.next(), Some("world"));
    }

    #[test]
    fn test_bin_view_array() {
        use arrow_array::Array;
        let mut builder = arrow_array::builder::BinaryViewBuilder::new();
        builder.append_value("hello");
        builder.append_null();
        builder.append_value("world");
        let array = builder.finish();

        let data = array.to_data();
        let arr = BinaryViewArray::from_data(&data);
        let mut x = arr.values_iter();
        assert_eq!(x.next(), Some("hello".as_bytes()));
        // TODO: Why is it not null
        assert_eq!(x.next(), Some("".as_bytes()));
        assert_eq!(x.next(), Some("world".as_bytes()));
    }
}
