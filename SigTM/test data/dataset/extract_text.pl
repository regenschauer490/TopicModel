use strict;
use warnings;
use Encode;
use utf8;
use Data::Dumper;
use MyUtility;

#
my $base_pass = "./user/timeline/";
my $src_pass = $base_pass . "raw" ."/";
#

binmode(STDOUT, ":utf8");

my @users = @{&GetFileList($src_pass)};

foreach my $u (@users){
	my @tweets = @{&SimpleReadDataFile($src_pass . $u, "cp932")};
	
	foreach my $tw (@tweets){
		my $tweet;
		eval{	
			$tweet = &JsonDecode($tw);
		};
		if($@){ print "json decode error\n"; next; }
		
		my $text = $tweet->{"text"};
		
		$text =~ s/\n/./g;
		&SimpleSaveAddDataFile($text."\n", $base_pass.encode("cp932",$u), "cp932" );
	}
}