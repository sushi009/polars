(function() {
    var type_impls = Object.fromEntries([["polars",[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-IntoScalar-for-u32\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/into_scalar.rs.html#28-56\">Source</a><a href=\"#impl-IntoScalar-for-u32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars/prelude/trait.IntoScalar.html\" title=\"trait polars::prelude::IntoScalar\">IntoScalar</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.into_scalar\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/into_scalar.rs.html#28-56\">Source</a><a href=\"#method.into_scalar\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars/prelude/trait.IntoScalar.html#tymethod.into_scalar\" class=\"fn\">into_scalar</a>(self, dtype: <a class=\"enum\" href=\"polars/prelude/enum.DataType.html\" title=\"enum polars::prelude::DataType\">DataType</a>) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"struct\" href=\"polars/prelude/struct.Scalar.html\" title=\"struct polars::prelude::Scalar\">Scalar</a>, <a class=\"enum\" href=\"polars/prelude/enum.PolarsError.html\" title=\"enum polars::prelude::PolarsError\">PolarsError</a>&gt;</h4></section></div></details>","IntoScalar","polars::prelude::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Literal-for-u32\" class=\"impl\"><a href=\"#impl-Literal-for-u32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars/prelude/trait.Literal.html\" title=\"trait polars::prelude::Literal\">Literal</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.lit\" class=\"method trait-impl\"><a href=\"#method.lit\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars/prelude/trait.Literal.html#tymethod.lit\" class=\"fn\">lit</a>(self) -&gt; <a class=\"enum\" href=\"polars/prelude/enum.Expr.html\" title=\"enum polars::prelude::Expr\">Expr</a></h4></section></summary><div class='docblock'><a href=\"polars/prelude/enum.Expr.html#variant.Literal\" title=\"variant polars::prelude::Expr::Literal\">Literal</a> expression.</div></details></div></details>","Literal","polars::prelude::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-NumericNative-for-u32\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/mod.rs.html#386\">Source</a><a href=\"#impl-NumericNative-for-u32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars/prelude/trait.NumericNative.html\" title=\"trait polars::prelude::NumericNative\">NumericNative</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h3></section></summary><div class=\"impl-items\"><section id=\"associatedtype.PolarsType\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/mod.rs.html#387\">Source</a><a href=\"#associatedtype.PolarsType\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars/prelude/trait.NumericNative.html#associatedtype.PolarsType\" class=\"associatedtype\">PolarsType</a> = <a class=\"struct\" href=\"polars/prelude/struct.UInt32Type.html\" title=\"struct polars::prelude::UInt32Type\">UInt32Type</a></h4></section><section id=\"associatedtype.TrueDivPolarsType\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/mod.rs.html#388\">Source</a><a href=\"#associatedtype.TrueDivPolarsType\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars/prelude/trait.NumericNative.html#associatedtype.TrueDivPolarsType\" class=\"associatedtype\">TrueDivPolarsType</a> = <a class=\"struct\" href=\"polars/prelude/struct.Float64Type.html\" title=\"struct polars::prelude::Float64Type\">Float64Type</a></h4></section></div></details>","NumericNative","polars::prelude::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-TakeExtremum-for-u32\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_core/frame/group_by/aggregations/mod.rs.html#281\">Source</a><a href=\"#impl-TakeExtremum-for-u32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars/prelude/aggregations/trait.TakeExtremum.html\" title=\"trait polars::prelude::aggregations::TakeExtremum\">TakeExtremum</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.take_min\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/frame/group_by/aggregations/mod.rs.html#281\">Source</a><a href=\"#method.take_min\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars/prelude/aggregations/trait.TakeExtremum.html#tymethod.take_min\" class=\"fn\">take_min</a>(self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h4></section><section id=\"method.take_max\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/frame/group_by/aggregations/mod.rs.html#281\">Source</a><a href=\"#method.take_max\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars/prelude/aggregations/trait.TakeExtremum.html#tymethod.take_max\" class=\"fn\">take_max</a>(self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h4></section></div></details>","TakeExtremum","polars::prelude::IdxSize"]]],["polars_core",[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-IntoScalar-for-u32\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/into_scalar.rs.html#28-56\">Source</a><a href=\"#impl-IntoScalar-for-u32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_core/datatypes/trait.IntoScalar.html\" title=\"trait polars_core::datatypes::IntoScalar\">IntoScalar</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.into_scalar\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/into_scalar.rs.html#28-56\">Source</a><a href=\"#method.into_scalar\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_core/datatypes/trait.IntoScalar.html#tymethod.into_scalar\" class=\"fn\">into_scalar</a>(self, dtype: <a class=\"enum\" href=\"polars_core/datatypes/enum.DataType.html\" title=\"enum polars_core::datatypes::DataType\">DataType</a>) -&gt; <a class=\"type\" href=\"polars_core/error/type.PolarsResult.html\" title=\"type polars_core::error::PolarsResult\">PolarsResult</a>&lt;<a class=\"struct\" href=\"polars_core/scalar/struct.Scalar.html\" title=\"struct polars_core::scalar::Scalar\">Scalar</a>&gt;</h4></section></div></details>","IntoScalar","polars_core::prelude::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-NumericNative-for-u32\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/mod.rs.html#386-389\">Source</a><a href=\"#impl-NumericNative-for-u32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_core/datatypes/trait.NumericNative.html\" title=\"trait polars_core::datatypes::NumericNative\">NumericNative</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h3></section></summary><div class=\"impl-items\"><section id=\"associatedtype.PolarsType\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/mod.rs.html#387\">Source</a><a href=\"#associatedtype.PolarsType\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars_core/datatypes/trait.NumericNative.html#associatedtype.PolarsType\" class=\"associatedtype\">PolarsType</a> = <a class=\"struct\" href=\"polars_core/datatypes/struct.UInt32Type.html\" title=\"struct polars_core::datatypes::UInt32Type\">UInt32Type</a></h4></section><section id=\"associatedtype.TrueDivPolarsType\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/datatypes/mod.rs.html#388\">Source</a><a href=\"#associatedtype.TrueDivPolarsType\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars_core/datatypes/trait.NumericNative.html#associatedtype.TrueDivPolarsType\" class=\"associatedtype\">TrueDivPolarsType</a> = <a class=\"struct\" href=\"polars_core/datatypes/struct.Float64Type.html\" title=\"struct polars_core::datatypes::Float64Type\">Float64Type</a></h4></section></div></details>","NumericNative","polars_core::prelude::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-TakeExtremum-for-u32\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_core/frame/group_by/aggregations/mod.rs.html#281\">Source</a><a href=\"#impl-TakeExtremum-for-u32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_core/frame/group_by/aggregations/trait.TakeExtremum.html\" title=\"trait polars_core::frame::group_by::aggregations::TakeExtremum\">TakeExtremum</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h3><span class=\"item-info\"><div class=\"stab portability\">Available on <strong>crate feature <code>algorithm_group_by</code></strong> only.</div></span></section></summary><div class=\"impl-items\"><section id=\"method.take_min\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/frame/group_by/aggregations/mod.rs.html#281\">Source</a><a href=\"#method.take_min\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_core/frame/group_by/aggregations/trait.TakeExtremum.html#tymethod.take_min\" class=\"fn\">take_min</a>(self, other: Self) -&gt; Self</h4></section><section id=\"method.take_max\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_core/frame/group_by/aggregations/mod.rs.html#281\">Source</a><a href=\"#method.take_max\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_core/frame/group_by/aggregations/trait.TakeExtremum.html#tymethod.take_max\" class=\"fn\">take_max</a>(self, other: Self) -&gt; Self</h4></section></div></details>","TakeExtremum","polars_core::prelude::IdxSize"]]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[6197,4929]}