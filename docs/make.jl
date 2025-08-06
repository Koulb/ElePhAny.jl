using Documenter, ElectronPhonon

makedocs(
    sitename="ElePhAny",
    authors="Aleksandr Poliukhin",
    clean=true,
    modules=[ElectronPhonon],
    checkdocs=:exports,
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Examples" => "examples.md",
        "Index" => "api.md",
        ]
)

deploydocs(
    repo = "github.com/Koulb/ElePhAny.jl.git",
)