using Documenter, ElectronPhonon

makedocs(
    sitename="ElePhAny.jl",
    authors="Aleksandr Poliukhin",
    clean=true,
    modules=[ElectronPhonon],
    checkdocs=:exports,
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md"
        ]
)
