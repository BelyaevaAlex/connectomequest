#!/usr/bin/env perl
use strict;
use warnings;

my $path = shift @ARGV or die "usage: $0 MAIN_PY\n";
open my $input, '<:raw', $path or die "cannot read $path: $!\n";
local $/;
my $source = <$input>;
close $input;
my $newline = $source =~ /\r\n/ ? "\r\n" : "\n";

if ($source !~ /parser\.add_argument\('--tag'/) {
    my $needle = "    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')" . $newline;
    my $replacement = $needle . "    parser.add_argument('--tag', default=None, type=str, help='stable run directory name')" . $newline;
    my $count = ($source =~ s/\Q$needle\E/$replacement/);
    die "checkpoint_path insertion point not found in $path\n" unless $count == 1;
}

if ($source !~ /cur_time = args\.tag or parse_time\(\)/) {
    my $needle = "    cur_time = parse_time()" . $newline;
    my $replacement = "    cur_time = args.tag or parse_time()" . $newline;
    my $count = ($source =~ s/\Q$needle\E/$replacement/);
    die "cur_time insertion point not found in $path\n" unless $count == 1;
}

if ($source !~ /CQ_CUDA_MEMORY_FRACTION/) {
    my $needle = "    if args.cuda:" . $newline . "        model = model.cuda()" . $newline;
    my $replacement = "    if args.cuda:" . $newline
        . "        memory_fraction = os.environ.get(\"CQ_CUDA_MEMORY_FRACTION\")" . $newline
        . "        if memory_fraction is not None:" . $newline
        . "            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device=0)" . $newline
        . "        model = model.cuda()" . $newline;
    my $count = ($source =~ s/\Q$needle\E/$replacement/);
    die "CUDA model insertion point not found in $path\n" unless $count == 1;
}

if ($source !~ /class NullSummaryWriter/) {
    my $needle = "def parse_args(args=None):";
    my $replacement = "class NullSummaryWriter:" . $newline
        . "    def add_scalar(self, *args, **kwargs):" . $newline
        . "        pass" . $newline . $newline
        . $needle;
    my $count = ($source =~ s/\Q$needle\E/$replacement/);
    die "parse_args insertion point not found in $path\n" unless $count == 1;
}

if ($source !~ /CQ_DISABLE_TENSORBOARD/) {
    my $needle = "    if not args.do_train: # if not training, then create tensorboard files in some tmp location" . $newline
        . "        writer = SummaryWriter(./logs-debug/unused-tb)" . $newline
        . "    else:" . $newline
        . "        writer = SummaryWriter(args.save_path)" . $newline;
    my $replacement = "    if os.environ.get(\"CQ_DISABLE_TENSORBOARD\") == \"1\":" . $newline
        . "        writer = NullSummaryWriter()" . $newline
        . "    elif not args.do_train: # if not training, then create tensorboard files in some tmp location" . $newline
        . "        writer = SummaryWriter(./logs-debug/unused-tb)" . $newline
        . "    else:" . $newline
        . "        writer = SummaryWriter(args.save_path)" . $newline;
    my $count = ($source =~ s/\Q$needle\E/$replacement/);
    die "SummaryWriter insertion point not found in $path\n" unless $count == 1;
}

open my $output, '>:raw', $path or die "cannot write $path: $!\n";
print {$output} $source;
close $output;
