using Documenter, ElectronPhonon

makedocs(
    sitename="ElePhAny",
    authors="Aleksandr Poliukhin",
    clean=true,
    modules=[ElectronPhonon],
    checkdocs=:exports,
    pages = [
        "Home" => "home.md",
        "Theory" => "theory.md",
        "Examples" => "examples.md",
        "Index" => "index.md",
        ]
)
