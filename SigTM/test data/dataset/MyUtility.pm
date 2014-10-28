package MyUtility;

use strict;
use warnings;
use JSON;
use Encode;
use utf8;
use Array::Utils qw(:all);
use Exporter;	# Exporterモジュールを使う

our @ISA    = qw(Exporter);				# Exporterモジュールを継承
our @EXPORT = qw(ListDrop ListTake ListDiff DoesExistFile GetFileList Print SimpleReadDataFile SimpleSaveAddDataFile SimpleSaveArrayDataFile SimpleSaveAddArrayDataFile JsonEncode JsonDecode);	# デフォルトでエクスポートするシンボル


# listの先頭からnum個を除去したlistを取得(元のlistは変更を加えない)
sub ListDrop($$)
{
	my $num = shift;
	my $list_ref = shift;
	my @result;
	my $size = scalar(@$list_ref);
	
	for(my $i=0; $i < $size; ++$i){
		if($i < $num){ next; }
		push(@result, $list_ref->[$i]);
	}
	
	return \@result;
}

# listの先頭からnum個分のlistを取得(元のlistは変更を加えない)
sub ListTake($$)
{
	my $num = shift;
	my $list_ref = shift;
	my @result;
	my $size = scalar(@$list_ref);
	
	for(my $i=0; $i < $num; ++$i){
		unless($i < $size){ last; }
		push(@result, $list_ref->[$i]);
	}
	
	return \@result;
}

# new かつ not prev な集合を求める
sub ListDiff($$)
{
	my $new_rf = shift;
	my $prev_rf = shift;
	
	my %grep;
	foreach my $prev (@$prev_rf){
		++$grep{$prev};
	}
	my @only_new = grep{ ++$grep{$_} < 2 } @$new_rf;
	
	return \@only_new;
}

#指定ファイルが存在するか
sub DoesExistFile($){
	my $filename = shift;
	
	return -f $filename;
}

#
sub GetFileList{
	my $dir = shift;
	my @result;
	opendir(DIRHANDLE, $dir);
	
	foreach(readdir(DIRHANDLE)){
		next if /^\.{1,2}$/;    # '.'や'..'をスキップ
		push(@result, $_);
	}
	closedir(DIRHANDLE);
	return \@result;
}

sub Print(){
	my $data = shift;
	my $mode = shift;
	$mode = "cp932" unless defined $mode;
	
	print encode($mode, $data);
}

sub SimpleReadDataFile($$){
	my $filename = shift;
	my $mcode = shift;
	my @contents;
	
	open(FH, "<", $filename) or die "- filehandle open error in SimpleReadDataFile :$! [$filename]\n";
	my $line;
	while($line = <FH>){
		$line = decode($mcode, $line);
		chomp($line);
		push(@contents, $line);
	}
	close(FH);
	
	return \@contents;
}

sub SimpleSaveAddDataFile($$$){
	my $data = shift;
	my $filename = shift;
	my $mcode = shift;
	
	open(FH , '>>' , $filename) or die "filehandle open error in SimpleSaveAddDataFile :$! [$filename]\n";
	#binmode(FH, ":encoding($mcode)");

	print FH encode($mcode, $data);

	close(FH);
}

sub SimpleSaveArrayDataFile($$$){
	my $data = shift;
	my $filename = shift;
	my $mcode = shift;
	
	open(FH , '>' , $filename) or die "filehandle open error in SimpleSaveArrayDataFile :$! [$filename]\n";
	#binmode(FH, ":encoding($mcode)");

	foreach my $e (@$data){
		print FH encode($mcode, $e."\n");
	}

	close(FH);
}

sub SimpleSaveAddArrayDataFile($$$){
	my $data = shift;
	my $filename = shift;
	my $mcode = shift;
	
	open(FH , '>>' , $filename) or die "filehandle open error in SimpleSaveArrayDataFile :$! [$filename]\n";
	#binmode(FH, ":encoding($mcode)");

	foreach my $e (@$data){
		print FH encode($mcode, $e."\n");
	}

	close(FH);
}

sub JsonEncode{	
	my $data = shift;
	my $savename = shift;
	my $option = shift;
	my $mcode = shift;
	
	my $json = JSON->new->encode($data);
	
	if(defined($savename)){
		open (FH, $option, $savename.'.txt') or die "- filehandle open error json encode :$!\n";
#		binmode(FH, ":utf8");
		print FH encode($mcode, $json);
		close (FH);
	}
	
	return $json;
}

sub JsonDecode{
	my $data = shift;
	
	if(!($data =~ m/^{/ or $data =~ m/^\[/)){
		print "not json response\n";
		return 0;
	}

	my $dec_json;
	if(utf8::is_utf8($data)){
	#	print "【utf8】";
		$dec_json = JSON->new->utf8(0)->decode($data);
	}
	else{
	#	print "【not utf8】";
		$dec_json = JSON->new->utf8(1)->decode($data);
	}
	
#	print Dumper %info;
	
	return $dec_json;
}

1;