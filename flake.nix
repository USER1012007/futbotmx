{
  description = "React frontend application";
  
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  
  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system:
        f {
          pkgs = import nixpkgs { inherit system; };
        }
      );
    in
    {
      devShells = forEachSystem ({ pkgs }:
        {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              python313Packages.ipython
            ];
          };
        }
      );
    };
}
