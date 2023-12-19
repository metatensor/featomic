use std::path::PathBuf;

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let generated_comment = "\
/* ============    Automatically generated file, DOT NOT EDIT.    ============ *
 *                                                                             *
 *    This file is automatically generated from the rascaline-c-api sources,   *
 *    using cbindgen. If you want to make change to this file (including       *
 *    documentation), make the corresponding changes in the rust sources.      *
 * =========================================================================== */";

    let mut config: cbindgen::Config = Default::default();
    config.language = cbindgen::Language::C;
    config.cpp_compat = true;
    config.includes = vec!["metatensor.h".into()];
    config.include_guard = Some("RASCALINE_H".into());
    config.include_version = false;
    config.documentation = true;
    config.documentation_style = cbindgen::DocumentationStyle::Doxy;
    config.header = Some(generated_comment.into());

    let result = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .map(|data| {
            let mut path = PathBuf::from("include");
            path.push("rascaline.h");
            data.write_to_file(&path);
        });

    if result.is_ok() {
        println!("cargo:rerun-if-changed=src");
    } else {
        // if rascaline header generation failed, we always re-run the build script
    }
}
