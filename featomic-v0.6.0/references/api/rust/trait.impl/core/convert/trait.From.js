(function() {
    var implementors = Object.fromEntries([["featomic",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"featomic/types/struct.Matrix3.html\" title=\"struct featomic::types::Matrix3\">Matrix3</a>&gt; for <a class=\"struct\" href=\"featomic/systems/struct.UnitCell.html\" title=\"struct featomic::systems::UnitCell\">UnitCell</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.83.0/alloc/boxed/struct.Box.html\" title=\"struct alloc::boxed::Box\">Box</a>&lt;dyn <a class=\"trait\" href=\"featomic/calculators/trait.CalculatorBase.html\" title=\"trait featomic::calculators::CalculatorBase\">CalculatorBase</a>&gt;&gt; for <a class=\"struct\" href=\"featomic/struct.Calculator.html\" title=\"struct featomic::Calculator\">Calculator</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.83.0/alloc/boxed/struct.Box.html\" title=\"struct alloc::boxed::Box\">Box</a>&lt;dyn <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/any/trait.Any.html\" title=\"trait core::any::Any\">Any</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>&gt;&gt; for <a class=\"enum\" href=\"featomic/enum.Error.html\" title=\"enum featomic::Error\">Error</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.83.0/core/str/error/struct.Utf8Error.html\" title=\"struct core::str::error::Utf8Error\">Utf8Error</a>&gt; for <a class=\"enum\" href=\"featomic/enum.Error.html\" title=\"enum featomic::Error\">Error</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://docs.rs/serde_json/1.0.133/serde_json/error/struct.Error.html\" title=\"struct serde_json::error::Error\">Error</a>&gt; for <a class=\"enum\" href=\"featomic/enum.Error.html\" title=\"enum featomic::Error\">Error</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;Error&gt; for <a class=\"enum\" href=\"featomic/enum.Error.html\" title=\"enum featomic::Error\">Error</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.f64.html\">f64</a>; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.array.html\">3</a>]&gt; for <a class=\"struct\" href=\"featomic/types/struct.Vector3D.html\" title=\"struct featomic::types::Vector3D\">Vector3D</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;[[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.f64.html\">f64</a>; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.array.html\">3</a>]; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.array.html\">3</a>]&gt; for <a class=\"struct\" href=\"featomic/types/struct.Matrix3.html\" title=\"struct featomic::types::Matrix3\">Matrix3</a>"]]],["metatensor",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.i32.html\">i32</a>&gt; for <a class=\"struct\" href=\"metatensor/struct.LabelValue.html\" title=\"struct metatensor::LabelValue\">LabelValue</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.isize.html\">isize</a>&gt; for <a class=\"struct\" href=\"metatensor/struct.LabelValue.html\" title=\"struct metatensor::LabelValue\">LabelValue</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.u32.html\">u32</a>&gt; for <a class=\"struct\" href=\"metatensor/struct.LabelValue.html\" title=\"struct metatensor::LabelValue\">LabelValue</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.83.0/std/primitive.usize.html\">usize</a>&gt; for <a class=\"struct\" href=\"metatensor/struct.LabelValue.html\" title=\"struct metatensor::LabelValue\">LabelValue</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.83.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.83.0/alloc/boxed/struct.Box.html\" title=\"struct alloc::boxed::Box\">Box</a>&lt;dyn <a class=\"trait\" href=\"metatensor/trait.Array.html\" title=\"trait metatensor::Array\">Array</a>&gt;&gt; for mts_array_t"]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[3721,1912]}