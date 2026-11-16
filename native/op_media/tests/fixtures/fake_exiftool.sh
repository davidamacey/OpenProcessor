#!/bin/sh
# A stand-in for `exiftool -stay_open True -@ -` that speaks its line protocol: one argument per
# line, `-execute<n>` runs the arguments read so far and ends the answer with `{ready<n>}`, and
# `-stay_open` then `False` makes it exit (https://exiftool.org/exiftool_pod.html). The pool tests
# start it through OP_EXIFTOOL_BIN, so they need no ExifTool.
#
# It answers with a one-row JSON list holding its process id ("Fake:Pid"), the number of the
# `-execute` that ended the request ("Fake:Execute") and the arguments it received ("Fake:Args"). The last argument is the path; its suffix selects a behaviour:
#   .hang      never answers (the process becomes a sleep)
#   .die       exits without answering
#   .diesonce  exits the first time a file of that name is read, answers afterwards
#   .noisy     writes a megabyte to standard error before answering
#   .slow      answers after 0.3 seconds
#   .empty     answers with nothing (as ExifTool does for a file that does not exist)
#   .badjson   answers with text that is not JSON
#   .big       answers with a 300 KB tag value (many reads of the pipe)
#   .huge      answers with 17 MiB, more than the pool accepts
# Started with other arguments than the pool's, it exits with 64.

[ "$*" = "-stay_open True -@ - -common_args -G -n" ] || exit 64

answer() {
    printf '[{\n"SourceFile": "%s",\n"Fake:Pid": %s,\n"Fake:Execute": %s,\n"Fake:Args": [%s]\n}]\n' "$1" "$$" "$id" "$2"
}

respond() {
    id=$1
    last=$2
    args=$3
    case "$last" in
    -ver) printf '12.76\n' ;;
    *.hang) exec sleep 300 ;;
    *.die) exit 3 ;;
    *.diesonce)
        if [ ! -e "$last.seen" ]; then
            : >"$last.seen"
            exit 3
        fi
        answer "$last" "$args"
        ;;
    *.noisy)
        head -c 1048576 /dev/zero >&2
        answer "$last" "$args"
        ;;
    *.slow)
        sleep 0.3
        answer "$last" "$args"
        ;;
    *.empty) ;;
    *.badjson) printf 'this is not json\n' ;;
    *.big)
        printf '[{"Fake:Pad": "'
        head -c 300000 /dev/zero | tr '\0' 'a'
        printf '"}]\n'
        ;;
    *.huge) head -c 17825792 /dev/zero | tr '\0' ' ' ;;
    *) answer "$last" "$args" ;;
    esac
    printf '{ready%s}\n' "$id"
}

args=""
last=""
while IFS= read -r line; do
    case "$line" in
    -stay_open)
        IFS= read -r _
        exit 0
        ;;
    -execute*)
        respond "${line#-execute}" "$last" "$args"
        args=""
        last=""
        ;;
    *)
        last=$line
        args="$args${args:+,}\"$line\""
        ;;
    esac
done
