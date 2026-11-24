# Sourced by the job scripts. Exports CMP_GIT_COMMIT, CMP_GIT_DIRTY and CMP_NOTEBOOK_SHA256 for the current folder.
# Run it where git works (the submission host); compute nodes may not have git, and then only inherit the values.
resolve_git_state() {
    command -v git > /dev/null || { echo "ERROR: git is not available on this host." >&2; return 1; }
    CMP_GIT_COMMIT=$(git rev-parse --verify HEAD) || { echo "ERROR: no Git commit could be resolved." >&2; return 1; }
    local changes
    changes=$(git status --porcelain --untracked-files=no) || { echo "ERROR: git status failed." >&2; return 1; }
    if [ -z "$changes" ]; then CMP_GIT_DIRTY=0; else CMP_GIT_DIRTY=1; fi
    CMP_NOTEBOOK_SHA256=$(sha256sum Cross_Market_Project.ipynb | cut -d' ' -f1)
    export CMP_GIT_COMMIT CMP_GIT_DIRTY CMP_NOTEBOOK_SHA256
}
