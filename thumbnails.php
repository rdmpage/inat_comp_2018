<?php

// Generate a list of folders, displaying one image from each

$basedir = dirname(__FILE__) . '/images';

$files1 = scandir($basedir);

echo '<html>
<head>
<style>

body {
	font-family:sans-serif;	
}

/* https://w3bits.com/css-masonry/ */	
.masonry { /* Masonry container */
  column-count: 6;
  column-gap: 1em;
  background-color: #eee;
  padding:1em;
}

.item { /* Masonry bricks or child elements */
  
  display: inline-block;
  margin: 0 0 1em;
  width: 100%;
  border: 1px solid #ddd;
  text-align:center;
  background-color: #fff;
}
</style>
</head>
<body>
<h1>Images of iNaturalist lepidoptera</h1>
<p>Each image is a single example from the corresponding folder.</p>
<div class="masonry">
';


foreach ($files1 as $dirname)
{
	if (preg_match('/^\d+$/', $dirname))
	{	
		echo '<div class="item">';
	
		$files2 = scandir($basedir . '/' . $dirname);
		
		echo '<img width="100" src="images/' . $dirname . '/' . $files2[2] . '">';
		
		echo '<br>' . $dirname;
		
		echo '</div>';
		
		// print_r($files2);
	}
} 

echo '</div>
</body>
</html>';

?>
