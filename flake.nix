{

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs, ... }:
    let
      # System types to support.
      supportedSystems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

      # Nixpkgs instantiated for supported system types.
      nixpkgsFor = forAllSystems (
        system:
        import nixpkgs {
          inherit system;
          overlays = [ self.overlays.default ];
          config = {
            allowUnfree = true;
          };
        }
      );
    in
    {

      formatter = forAllSystems (system: nixpkgsFor.${system}.nixfmt);

      overlays.default =
        final: prev:
        let
          pkgs = final.pkgs;
        in
        {
          devShell =
            let
              python-with-packages = pkgs.python3.withPackages (
                p: with p; [
                  autopep8
                  black
                  dayplot
                  fastapi
                  httpx
                  isort
                  pandas
                  pip
                  pylint
                  python-multipart
                  setuptools
                  uvicorn
                ]
              );
            in
            pkgs.mkShell {
              buildInputs = with pkgs; [
                nixfmt
                pre-commit
                python-with-packages
              ];
              shellHook = ''
                if [[ -z $using_direnv ]]; then
                  # print information about the development shell
                  echo "---------------------------------------------------------------------"
                  echo "How to use this Nix development shell:"
                  echo "python interpreter: ${python-with-packages}/bin/python3"
                  echo "python site packages: ${python-with-packages}/${python-with-packages.sitePackages}"
                  echo "---------------------------------------------------------------------"
                  echo "In case you need to set the PYTHONPATH environment variable, run:"
                  echo "export PYTHONPATH=${python-with-packages}/${python-with-packages.sitePackages}"
                  echo "---------------------------------------------------------------------"
                fi
              '';
            };
          python3 = prev.python3.override {
            packageOverrides = python-self: python-super: {
              dayplot = python-super.buildPythonPackage rec {
                pname = "dayplot";
                version = "0.4.2";
                pyproject = true;
                src = pkgs.fetchFromGitHub {
                  owner = "y-sunflower";
                  repo = "dayplot";
                  rev = "v${version}";
                  hash = "sha256-tmGuJhQzDP50ZtlD2odhxtugeLKKcyHBTAddlFnyCXs=";
                };
                build-system = with prev.python3.pkgs; [
                  setuptools
                  setuptools-scm
                ];
                dependencies = with prev.python3.pkgs; [
                  matplotlib
                  narwhals
                ];
                pythonImportsCheck = [
                  "dayplot"
                ];
                meta = with prev.lib; {
                  description = "Calendar heatmaps made super simple and highly customizable";
                  homepage = "https://github.com/y-sunflower/dayplot";
                  license = licenses.mit;
                  maintainers = with maintainers; [ MayNiklas ];
                  mainProgram = "dayplot";
                };
              };
            };
          };
        };

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgsFor.${system};
        in
        {
          default = pkgs.devShell;
        }
      );

    };
}
